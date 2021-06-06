from tsim.dpl_cmn import *
from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from ..util import CreatePredictionLog, CurrentPredict, Rmodel


setup_path = 'mysim.curriculum.tasks_domain.adhoc_amp.pouring.setup'
AMP_DTHETA2 = 50.
AMP_SMSZ = 10.
AMP_SHAKE_RANGE = 10.


def ObserveXSSA(l,xs_prev,keys):
  if any(key in keys for key in ('ps_rcv','dps_rcv','lp_pour','lp_pour_trg')):
    ps_rcv= Get4RcvEdgePoints(l, GPoseToX(l.sensors.x_rcv))
  xs= {}
  for key in keys:
    if key=='ps_rcv':  #4 edge point positions (x,y,z)*4 of receiver
      xs[key]= SSA(ps_rcv)
    elif key=='gh_abs':  #Gripper height (absolute value)
      xs[key]= SSA([l.config.GripperHeight])
    elif key=='dps_rcv':  #Displacement of ps_rcv from previous time
      xs[key]= SSA([math.atan(p1-p2) for p1,p2 in
                    zip(ps_rcv, ToList(xs_prev['ps_rcv'].X))])
    elif key=='v_rcv':  #Velocity norm of receiver
      xs[key]= SSA([math.atan(Norm(l.filtered.v_rcv))])
    elif key=='p_pour':  #Pouring axis position (x,y,z)
      xs[key]= SSA([p_pour for p_pour in l.sensors.p_pour])
    elif key=='p_pour_z':  #Pouring axis position (z)
      xs[key]= SSA([l.sensors.p_pour[2]])
    elif key=='lp_pour':  #Pouring axis position (x,y,z) in receiver frame
      xs[key]= SSA([p_pour-pc_rcv for p_pour,pc_rcv in
                    zip(l.sensors.p_pour,
                        np.array(ps_rcv).reshape(4,3).mean(axis=0)  #Center of ps_rcv
                        )])
    elif key=='lp_pour_trg':  #Pouring axis target position (x,y,z) in receiver frame
      xs[key]= SSA([p_pour-pc_rcv for p_pour,pc_rcv in
                    zip(l.p_pour_trg,
                        np.array(ps_rcv).reshape(4,3).mean(axis=0)  #Center of ps_rcv
                        )])
    elif key=='p_flow':  #Flow position (x,y)
      xs[key]= SSA([term_flow_center for term_flow_center in l.filtered.term_flow_center[:2]])
    elif key=='lp_flow':  #Flow position (x,y) in receiver frame
      xs[key]= SSA([math.atan(0.5*(term_flow_center-pc_rcv))
                    for term_flow_center, pc_rcv in
                    zip(l.filtered.term_flow_center[:2],
                        np.array(ps_rcv).reshape(4,3).mean(axis=0)  #Center of ps_rcv
                        )])
    elif key=='lp_flow2':  #Flow position (x,y) in receiver frame (no atan)
      xs[key]= SSA([term_flow_center-pc_rcv
                    for term_flow_center, pc_rcv in
                    zip(l.filtered.term_flow_center[:2],
                        np.array(ps_rcv).reshape(4,3).mean(axis=0)  #Center of ps_rcv
                        )])
    elif key=='lpp_flow':  #Flow position (x,y) relative to previous (before flowctrl) p_pour
      xs[key]= SSA([term_flow_center-p_pour
                    for term_flow_center, p_pour in zip(l.filtered.term_flow_center[:2],ToList(xs_prev['p_pour'].X)[:2])])
    elif key=='flow_var':  #Variance of flow
      xs[key]= SSA([2.0*math.sqrt(l.filtered.term_flow_var)])
    elif key=='a_pour':  #Amount poured in receiver
      xs[key]= SSA([l.filtered.amount])  #==0.0055*l.sensors.num_rcv
    elif key=='a_spill':  #Amount spilled out
      xs[key]= SSA([0.0 if l.filtered.num_spill<1 else -math.atan(0.5*l.filtered.num_spill)])
    elif key=='a_spill2':  #Amount spilled out
      xs[key]= SSA([0.1*l.filtered.num_spill])
    elif key=='a_total':  #Total amount moved from source
      xs[key]= SSA([0.0055*(l.config.BallNum-l.sensors.num_src)])
    elif key=='a_trg':  #Target amount
      xs[key]= SSA([l.amount_trg])
    elif key=='da_pour':  #Amount poured in receiver (displacement)
      xs[key]= SSA([l.filtered.amount - xs_prev['a_pour'].X[0,0]])
    elif key=='da_spill':  #Amount spilled out (displacement)
      xs[key]= SSA([(0.0 if l.filtered.num_spill<1 else -math.atan(0.5*l.filtered.num_spill))
                    - xs_prev['a_spill'].X[0,0]])
    elif key=='da_spill2':  #Amount spilled out (displacement)
      xs[key]= SSA([0.1*l.filtered.num_spill - xs_prev['a_spill2'].X[0,0]])
    elif key=='da_total':  #Total amount moved from source (displacement)
      xs[key]= SSA([0.0055*(l.config.BallNum-l.sensors.num_src) - xs_prev['a_total'].X[0,0]])
    elif key=='da_trg':  #Target amount (displacement)
      if 'amount' in l.filtered:
        xs[key]= SSA([max(0.0, l.amount_trg - l.filtered.amount)])
      else:
        xs[key]= SSA([max(0.0, l.amount_trg)])
    elif key=='size_srcmouth':  #Size of mouth of the source container
      xs[key]= SSA([l.config.SrcSize2H*AMP_SMSZ])
    elif key=='material':  #Material property (e.g. viscosity)
      xs[key]= SSA([l.config.ContactBounce, l.config.ContactBounceVel,
                    l.config.ViscosityParam1, l.config.ViscosityMaxDist])
    elif key=='material2':  #Material property (e.g. viscosity)
      xs[key]= SSA([l.config.ContactBounce, l.config.ContactBounceVel,
                    l.config.ViscosityParam1*1.0e6, l.config.ViscosityMaxDist])
  return xs


def Delta1(dim, s):
    assert(abs(s-int(s)) < 1.0e-6)
    p = [0.0]*dim
    p[int(s)] = 1.0
    return p


def Domain():  # SpaceDefs and Models (reward function) will be modified by curriculum. (For example, 'action' -> 'state', Rdamount -> Rdaspill, and so on.)
    domain = TGraphDynDomain()
    SP = TCompSpaceDef
    domain.SpaceDefs = {
        'skill': SP('select', num=2),  # Skill selection
        'ps_rcv': SP('state', 12),  # 4 edge point positions (x,y,z)*4 of receiver
        'gh_ratio': SP('state', 1, min=[0.0], max=[1.0]),  # Gripper height (ratio)
        'gh_abs': SP('state', 1),  # Gripper height (absolute value)
        'p_pour_trg0': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),  # Target pouring axis position of preparation before pouring (x,z)
        # NOTE: we stopped to plan p_pour_trg0
        'p_pour_trg': SP('action', 2, min=[0.2, 0.1], max=[1.2, 0.7]),  # Target pouring axis position (x,z)
        'dtheta1': SP('action', 1, min=[0.01], max=[0.02]),  # Pouring skill parameter for all skills
        'dtheta2': SP('action', 1, min=[0.002*AMP_DTHETA2], max=[0.02*AMP_DTHETA2]),  # Pouring skill parameter for 'tip'
        'shake_spd': SP('action', 1, min=[0.5], max=[1.2]),  # Pouring skill parameter for 'shake'
        'shake_range': SP('action', 1, min=[0.05*AMP_SHAKE_RANGE], max=[0.12*AMP_SHAKE_RANGE]),
        'shake_angle': SP('action', 1, min=[-0.5*math.pi], max=[0.5*math.pi]),
        # 'shake_axis2': SP('action',2,min=[0.05,-0.5*math.pi],max=[0.1,0.5*math.pi]),
        'p_pour': SP('state', 3),  # Pouring axis position (z)
        # 'p_pour_z': SP('state', 1),  # Pouring axis position (z)
        'lp_pour': SP('state', 3),  # Pouring axis position (x,y,z) in receiver frame
        'dps_rcv': SP('state', 12),  # Displacement of ps_rcv from previous time
        'v_rcv': SP('state', 1),  # Velocity norm of receiver
        # 'p_flow': SP('state',2),  #Flow position (x,y)
        'lp_flow': SP('state', 2),  # Flow position (x,y) in receiver frame
        # 'lpp_flow': SP('state', 2),  # Flow position (x,y) relative to previous (before flowctrl) p_pour
        'flow_var': SP('state', 1),  # Variance of flow
        'a_pour': SP('state', 1),  # Amount poured in receiver
        'a_spill2': SP('state', 1),  # Amount spilled out
        'a_total':  SP('state', 1),  # Total amount moved from source
        'a_trg': SP('state', 1),  # Target amount
        'da_pour': SP('state', 1),  # Amount poured in receiver (displacement)
        'da_spill2': SP('state', 1),  # Amount spilled out (displacement)
        'da_total':  SP('state', 1),  # Total amount moved from source (displacement)
        'da_trg': SP('state', 1),  # Target amount (displacement)
        'size_srcmouth': SP('state', 1),  # Size of mouth of the source container
        'material2': SP('state', 4),  # Material property (e.g. viscosity)
        REWARD_KEY:  SP('state', 1),
    }
    domain.Models = {
        # key:[In,Out,F],
        'Fnone': [[], [], None],
        'Fgrasp': [['gh_ratio'], ['gh_abs'], None],  # Grasping. NOTE: removed ps_rcv
        'Fmvtorcv': [  # Move to receiver
            ['ps_rcv', 'gh_abs', 'p_pour', 'p_pour_trg0'],
            ['ps_rcv', 'p_pour'], None],
        'Fmvtorcv_rcvmv': [  # Move to receiver: receiver movement
            ['ps_rcv', 'gh_abs', 'p_pour', 'p_pour_trg0'],
            ['dps_rcv', 'v_rcv'], None],
        'Fmvtopour2': [  # Move to pouring point
            ['gh_abs', 'p_pour_trg'],
            ['lp_pour'], None],
        'Ftip': [  # Flow control with tip.
            ['gh_abs', 'lp_pour',  # Removed 'p_pour_trg0','p_pour_trg'
             'da_trg', 'size_srcmouth', 'material2',
             'dtheta1', 'dtheta2'],
            ['da_total', 'lp_flow', 'flow_var'], None],
        'Fshake': [  # Flow control with shake.
            ['gh_abs', 'lp_pour',  # Removed 'p_pour_trg0','p_pour_trg'
             'da_trg', 'size_srcmouth', 'material2',
             'dtheta1', 'shake_spd', 'shake_range', 'shake_angle'],
            ['da_total', 'lp_flow', 'flow_var'], None],
        'Famount': [  # Amount model common for tip and shake
            ['lp_pour',  # Removed 'gh_abs','p_pour_trg0','p_pour_trg'
             'da_trg', 'material2',  # Removed 'size_srcmouth'
             'da_total', 'lp_flow', 'flow_var'],
            ['da_pour', 'da_spill2'], None],
        "Rdapour_gentle": [['da_trg', 'da_pour'], [REWARD_KEY], Rmodel("Fdapour_gentle")],
        "Rdaspill": [["da_spill2"], [REWARD_KEY], Rmodel("Fdaspill")],
        'P1': [[], [PROB_KEY], TLocalLinear(0, 1, lambda x:[1.0], lambda x:[0.0])],
        'P2':  [[], [PROB_KEY], TLocalLinear(0, 2, lambda x:[1.0]*2, lambda x:[0.0]*2)],
        'Pskill': [['skill'], [PROB_KEY], TLocalLinear(0, 2, lambda s:Delta1(2, s[0]), lambda s:[0.0]*2)],
    }
    domain.Graph = {
        'n0': TDynNode(None, 'P1', ('Fgrasp', 'n1')),
        'n1': TDynNode('n0', 'P1', ('Fnone', 'n2a')),
        'n2a': TDynNode('n1', 'P1', ('Fmvtopour2', 'n2b')),
        'n2b': TDynNode('n2a', 'P1', ('Fnone', 'n2c')),
        'n2c': TDynNode('n2b', 'Pskill', ('Ftip', 'n3ti'), ('Fshake', 'n3sa')),
        # Tipping:
        'n3ti': TDynNode('n2c', 'P1', ('Famount', 'n4ti')),
        'n4ti': TDynNode('n3ti', 'P2', ('Rdapour_gentle', 'n4tir1'), ('Rdaspill', 'n4tir2')),
        'n4tir1': TDynNode('n4ti'),
        'n4tir2': TDynNode('n4ti'),
        # Shaking-A:
        'n3sa': TDynNode('n2c', 'P1', ('Famount', 'n4sa')),
        'n4sa': TDynNode('n3sa', 'P2', ('Rdapour_gentle', 'n4sar1'), ('Rdaspill', 'n4sar2')),
        'n4sar1': TDynNode('n4sa'),
        'n4sar2': TDynNode('n4sa'),
    }

    return domain


def ConfigCallback(ct, l, sim):  # This will be modified by task's setup. (For example, l.custom_mtr -> "natto", l.custom_smsz -> 0.055, and so on.)
    m_setup = ct.Load(setup_path)
    l.amount_trg = 0.3
    # Note: In this subtask, we do not use IsSpilled and IsPoured
    # l.spilled_stop = 10
    l.config.RcvPos = [0.6, l.config.RcvPos[1], l.config.RcvPos[2]]
    # l.config.RcvPos= [0.8+0.6*(random.random()-0.5), l.config.RcvPos[1], l.config.RcvPos[2]]
    CPrint(3, 'l.config.RcvPos=', l.config.RcvPos)
    for key, value in l.opt_conf['config'].iteritems():
        setattr(l.config, key, value)

    if l.rcv_size == 'static':
        l.config.RcvSize = [0.3, 0.4, 0.2]
    elif l.rcv_size == 'random':
        rsx = Rand(0.25, 0.5)
        rsy = Rand(0.1, 0.2)/rsx
        rsz = Rand(0.2, 0.5)
        l.config.RcvSize = [rsx, rsy, rsz]

    if l.mtr_smsz == 'curriculum_test':
        m_setup.SetMaterial(l, preset=('nobounce', 'ketchup')[RandI(2)])
        l.config.SrcSize2H = Rand(0.03, 0.08)
    elif l.mtr_smsz == 'nobounce_large':
        m_setup.SetMaterial(l, preset='nobounce')
        l.config.SrcSize2H = Rand(0.03, 0.055)
    elif l.mtr_smsz == 'ketchup_small':
        m_setup.SetMaterial(l, preset='ketchup')
        l.config.SrcSize2H = Rand(0.055, 0.08)
    else:
        raise(Exception("l.mtr_smsz is '"+l.mtr_smsz+"', but this is invalid name."))
    CPrint(3, 'l.config.ViscosityParam1=', l.config.ViscosityParam1)
    CPrint(3, 'l.config.SrcSize2H=', l.config.SrcSize2H)


def Execute(ct, l):
    l.node_best_tree = []
    l.pred_true_log = []
    l.user_viz = []  # Use in dpl_cmn

    l.dpl.NewEpisode()
    try:
        ct.Run(setup_path, l)
        sim = ct.sim

        actions = {
            'grab': lambda a: ct.Run('mysim.act.grab_sv', a),
            'move_to_rcv': lambda a: ct.Run('mysim.act.move_to_rcv_sv', a),
            'move_to_pour': lambda a: ct.Run('mysim.act.move_to_pour_sv', a),
            'tip': lambda a: ct.Run('mysim.act.tip', a),
            'shake': lambda a: ct.Run('mysim.act.shake', a),
        }

        obs_keys0 = (
            'ps_rcv',
            'p_pour',
            'lp_pour',
            'a_trg',
            'size_srcmouth',
            'material2',
        )
        obs_keys_after_grab = obs_keys0+('gh_abs',)
        obs_keys_before_flow = obs_keys_after_grab + \
            (
                'a_pour',
                'a_spill2',
                'a_total',
            )
        obs_keys_after_flow = obs_keys_before_flow + \
            (
                'lp_flow',
                'flow_var',
                'da_pour',
                'da_spill2',
                'da_total',
            )

        l.xs = TContainer()  # l.xs.NODE= XSSA
        l.idb = TContainer()  # l.idb.NODE= index in DB

        ############################################################################
        # n0: Initialize skill params, Plan, Execute grab
        ############################################################################
        CPrint(2, 'Node:', 'n0')
        l.xs.n0 = ObserveXSSA(l, None, obs_keys0+('da_trg',))
        # Heuristic init guess
        pc_rcv = np.array(l.xs.n0['ps_rcv'].X).reshape(4, 3).mean(axis=0)  # Center of ps_rcv
        l.xs.n0['p_pour_trg0'] = SSA(Vec([-0.3, 0.35])+Vec([pc_rcv[0], pc_rcv[2]]))
        l.xs.n0['gh_ratio'] = SSA([0.5])
        # l.xs.n0['p_pour_trg'] = SSA(Vec([Rand(0.2, 1.2), Rand(0.1, 0.7)]))
        l.xs.n0['dtheta1'] = SSA([0.014])
        # l.xs.n0['dtheta2'] = SSA([0.004])
        # l.xs.n0['shake_spd'] = SSA([0.8])
        # l.xs.n0['shake_range'] = SSA([0.08])
        # l.xs.n0['shake_angle'] = SSA([0.0])
        res = l.dpl.Plan('n0', l.xs.n0, l.interactive)
        if l.pour_skill == "tip":
            l.xs.n0['skill'] = SSA([0])
        elif l.pour_skill == "shake":
            l.xs.n0['skill'] = SSA([1])
        l.node_best_tree.append(res.PTree)
        l.idb.n0 = l.dpl.DB.AddToSeq(parent=None, name='n0', xs=l.xs.n0)
        l.xs.prev = l.xs.n0
        l.idb.prev = l.idb.n0
        gh_ratio = ToList(l.xs.n0['gh_ratio'].X)[0]
        actions['grab']({'gh_ratio': gh_ratio})

        ############################################################################
        # n1: Update Fgrasp, Plan (option), Execute move_to_rcv
        ############################################################################
        CPrint(2, 'Node:', 'n1')
        l.xs.n1 = CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n1, ObserveXSSA(l, l.xs.prev, obs_keys_after_grab))
        CreatePredictionLog(l, "Fgrasp", l.xs.prev, l.xs.n1)
        l.dpl.MM.Models['Fgrasp'][2].Options.update(l.nn_options)
        l.dpl.MM.Update('Fgrasp', l.xs.prev, l.xs.n1, not_learn=l.not_learn)
        if "n1" in l.planning_node:
            res = l.dpl.Plan('n1', l.xs.n1)
            l.node_best_tree.append(res.PTree)
        l.idb.n1 = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n1', xs=l.xs.n1)
        l.xs.prev = l.xs.n1
        l.idb.prev = l.idb.n1
        p_pour_trg0 = ToList(l.xs.n1['p_pour_trg0'].X)
        p_pour_trg = ToList(l.xs.n1['p_pour_trg'].X)  # l.xs.n0['p_pour_trg'].X?
        actions['move_to_rcv']({'p_pour_trg0': p_pour_trg0})
        VizPP(l, [p_pour_trg0[0], 0.0, p_pour_trg0[1]], [0., 1., 0.])
        VizPP(l, [p_pour_trg[0], 0.0, p_pour_trg[1]], [0.5, 0., 1.])

        ############################################################################
        # n1rcvmv: Update Fmvtorcv_rcvmv, Plan (option),
        ############################################################################
        CPrint(2, 'Node:', 'n1rcvmv')
        l.xs.n1rcvmv = CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n1rcvmv, ObserveXSSA(l, l.xs.prev, ('dps_rcv', 'v_rcv')))
        CreatePredictionLog(l, "Fmvtorcv_rcvmv", l.xs.prev, l.xs.n1rcvmv)
        l.dpl.MM.Models['Fmvtorcv_rcvmv'][2].Options.update(l.nn_options)
        l.dpl.MM.Update('Fmvtorcv_rcvmv', l.xs.prev, l.xs.n1rcvmv, not_learn=l.not_learn)
        if "n1rcvmv" in l.planning_node:
            res = l.dpl.Plan('n1rcvmv', l.xs.n1rcvmv)
            l.node_best_tree.append(res.PTree)
        l.idb.n1rcvmv = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n1rcvmv', xs=l.xs.n1rcvmv)

        ############################################################################
        # n2a: Update Fmvtorcv
        ############################################################################
        CPrint(2, 'Node:', 'n2a')
        l.xs.n2a = CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n2a, ObserveXSSA(l, l.xs.prev, obs_keys_after_grab))
        CreatePredictionLog(l, "Fmvtorcv", l.xs.prev, l.xs.n2a)
        l.dpl.MM.Models['Fmvtorcv'][2].Options.update(l.nn_options)
        l.dpl.MM.Update('Fmvtorcv', l.xs.prev, l.xs.n2a, not_learn=l.not_learn)

        repeated = False  # For try-and-error learning
        while True:  # Try-and-error starts from here.
            ############################################################################
            # n2a: Plan (option), Execute move_to_pour
            ############################################################################
            CPrint(2, 'Node:', 'n2a')
            l.xs.n2a = CopyXSSA(l.xs.prev)
            if repeated:
                # Delete actions and selections (e.g. skill) to plan again from initial guess.
                for key in l.xs.n2a.keys():
                    if l.dpl.d.SpaceDefs[key].Type in ('action', 'select'):
                        del l.xs.n2a[key]
            InsertDict(l.xs.n2a, ObserveXSSA(l, l.xs.prev, obs_keys_after_grab+('da_trg',)))
            if "n2a" in l.planning_node:
                res = l.dpl.Plan('n2a', l.xs.n2a, l.interactive)
                l.node_best_tree.append(res.PTree)
            l.idb.n2a = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n2a', xs=l.xs.n2a)
            l.xs.prev = l.xs.n2a
            l.idb.prev = l.idb.n2a
            # Execute move_to_pour
            p_pour_trg = ToList(l.xs.n2a['p_pour_trg'].X)
            actions['move_to_pour']({'p_pour_trg': p_pour_trg})
            l.user_viz.pop()
            VizPP(l, [p_pour_trg[0], 0.0, p_pour_trg[1]], [1., 0., 1.])

            ############################################################################
            # n2b: Update Fmvtopour2
            ############################################################################
            CPrint(2, 'Node:', 'n2b')
            l.xs.n2b = CopyXSSA(l.xs.prev)
            InsertDict(l.xs.n2b, ObserveXSSA(l, l.xs.prev, obs_keys_after_grab))
            CreatePredictionLog(l, "Fmvtopour2", l.xs.prev, l.xs.n2b)
            l.dpl.MM.Models['Fmvtopour2'][2].Options.update(l.nn_options)
            l.dpl.MM.Update('Fmvtopour2', l.xs.prev, l.xs.n2b, not_learn=l.not_learn)
            if "n2b" in l.planning_node:
                res = l.dpl.Plan('n2b', l.xs.n2b, l.interactive)
                l.node_best_tree.append(res.PTree)
            l.idb.n2b = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n2b', xs=l.xs.n2b)
            l.xs.prev = l.xs.n2b
            l.idb.prev = l.idb.n2b

            ############################################################################
            # n2c: Execute pouring skill (Selection is already planned before node)
            ############################################################################
            CPrint(2, 'Node:', 'n2c')
            l.xs.n2c = CopyXSSA(l.xs.prev)
            InsertDict(l.xs.n2c, ObserveXSSA(l, l.xs.prev, obs_keys_before_flow))
            l.idb.n2c = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n2c', xs=l.xs.n2c)
            l.xs.prev = l.xs.n2c
            l.idb.prev = l.idb.n2c

            idx = int(l.xs.n2c['skill'].X[0])
            selected_skill = ('tip', 'shake')[idx]

            if selected_skill == 'tip':
                dtheta1 = l.xs.n2c['dtheta1'].X[0, 0]
                dtheta2 = l.xs.n2c['dtheta2'].X[0, 0]
                actions['tip']({'dtheta1': dtheta1, 'dtheta2': dtheta2/AMP_DTHETA2})

                ############################################################################
                # n3ti: Update Ftip
                ############################################################################
                CPrint(2, 'Node:', 'n3ti')
                l.xs.n3ti = CopyXSSA(l.xs.prev)
                InsertDict(l.xs.n3ti, ObserveXSSA(l, l.xs.prev, obs_keys_after_flow))
                xs_in = CopyXSSA(l.xs.prev)
                xs_in['lp_pour'] = l.xs.n2c['lp_pour']
                CreatePredictionLog(l, "Ftip", xs_in, l.xs.n3ti)
                l.dpl.MM.Models['Ftip'][2].Options.update(l.nn_options)
                l.dpl.MM.Update('Ftip', xs_in, l.xs.n3ti, not_learn=l.not_learn)
                if "n3ti" in l.planning_node:
                    res = l.dpl.Plan('n3ti', l.xs.n3ti, l.interactive)
                    l.node_best_tree.append(res.PTree)
                l.idb.n3ti = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n3ti', xs=l.xs.n3ti)
                l.xs.prev = l.xs.n3ti
                l.idb.prev = l.idb.n3ti

                ############################################################################
                # n4ti: Update Famount
                ############################################################################
                CPrint(2, 'Node:', 'n4ti')
                l.xs.n4ti = CopyXSSA(l.xs.prev)
                InsertDict(l.xs.n4ti, ObserveXSSA(l, l.xs.prev, ()))  # Observation is omitted since there is no change
                xs_in = CopyXSSA(l.xs.prev)
                xs_in['lp_pour'] = l.xs.n2c['lp_pour']
                CreatePredictionLog(l, "Famount", xs_in, l.xs.n4ti)
                l.dpl.MM.Models['Famount'][2].Options.update(l.nn_options)
                l.dpl.MM.Update('Famount', xs_in, l.xs.n4ti, not_learn=l.not_learn)

                l.idb.n4ti = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n4ti', xs=l.xs.n4ti)
                l.xs.prev = l.xs.n4ti
                l.idb.prev = l.idb.n4ti
                
                ############################################################################
                # n4tir1: Calculate Rdapour_gentle
                ############################################################################
                CPrint(2, 'Node:', 'n4tir1')
                l.xs.n4tir1 = l.dpl.Forward('Rdapour_gentle', l.xs.prev)
                l.idb.n4tir1 = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n4tir1', xs=l.xs.n4tir1)
                
                ############################################################################
                # n4tir2: Calculate Rdaspill
                ############################################################################
                CPrint(2, 'Node:', 'n4tir2')
                l.xs.n4tir2 = l.dpl.Forward('Rdaspill', l.xs.prev)
                l.idb.n4tir2 = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n4tir2', xs=l.xs.n4tir2)

            elif selected_skill == 'shake':
                dtheta1 = l.xs.n2c['dtheta1'].X[0, 0]
                shake_spd = l.xs.n2c['shake_spd'].X[0, 0]
                # shake_axis2 = ToList(l.xs.n2c['shake_axis2'].X)
                shake_axis2 = ToList([l.xs.n2c['shake_range'].X.item()/AMP_SHAKE_RANGE, l.xs.n2c['shake_angle'].X.item()])
                actions['shake']({'dtheta1': dtheta1, 'shake_spd': shake_spd, 'shake_axis2': shake_axis2})

                ############################################################################
                # n3sa: Update Fshake
                ############################################################################
                CPrint(2, 'Node:', 'n3sa')
                l.xs.n3sa = CopyXSSA(l.xs.prev)
                InsertDict(l.xs.n3sa, ObserveXSSA(l, l.xs.prev, obs_keys_after_flow))
                xs_in = CopyXSSA(l.xs.prev)
                xs_in['lp_pour'] = l.xs.n2c['lp_pour']
                CreatePredictionLog(l, "Fshake", xs_in, l.xs.n3sa)
                l.dpl.MM.Models['Fshake'][2].Options.update(l.nn_options)
                l.dpl.MM.Update('Fshake', xs_in, l.xs.n3sa, not_learn=l.not_learn)
                if "n3sa" in l.planning_node:
                    res = l.dpl.Plan('n3sa', l.xs.n3sa, l.interactive)
                    l.node_best_tree.append(res.PTree)
                l.idb.n3sa = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n3sa', xs=l.xs.n3sa)
                l.xs.prev = l.xs.n3sa
                l.idb.prev = l.idb.n3sa

                ############################################################################
                # n4sa: Update Famount
                ############################################################################
                CPrint(2, 'Node:', 'n4sa')
                l.xs.n4sa = CopyXSSA(l.xs.prev)
                InsertDict(l.xs.n4sa, ObserveXSSA(l, l.xs.prev, ()))  # Observation is omitted since there is no change
                # WARNING:NOTE: Famount uses 'lp_pour' as input, so here we use a trick:
                xs_in = CopyXSSA(l.xs.prev)
                xs_in['lp_pour'] = l.xs.n2c['lp_pour']
                CreatePredictionLog(l, "Famount", xs_in, l.xs.n4sa)
                l.dpl.MM.Models['Famount'][2].Options.update(l.nn_options)
                l.dpl.MM.Update('Famount', xs_in, l.xs.n4sa, not_learn=l.not_learn)
                l.idb.n4sa = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n4sa', xs=l.xs.n4sa)
                l.xs.prev = l.xs.n4sa
                l.idb.prev = l.idb.n4sa

                ############################################################################
                # n4sar1: Caluculate Rdapour_gentle
                ############################################################################
                CPrint(2, 'Node:', 'n4sar1')
                l.xs.n4sar1 = l.dpl.Forward('Rdapour_gentle', l.xs.prev)
                l.idb.n4sar1 = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n4sar1', xs=l.xs.n4sar1)
                
                ############################################################################
                # n4sar2: Caluculate Rdaspill
                ############################################################################
                CPrint(2, 'Node:', 'n4sar2')
                l.xs.n4sar2 = l.dpl.Forward('Rdaspill', l.xs.prev)
                l.idb.n4sar2 = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n4sar2', xs=l.xs.n4sar2)

            if "n2" in l.planning_node:
                # Conditions to break the try-and-error loop
                if l.IsFlowedOut():
                    break
                if l.IsTimeout() or l.IsEmpty():  # or l.IsSpilled()
                    break
                if not IsSuccess(l.exec_status):
                    break
                repeated = True
            else:
                break
    finally:
        ct.sim.StopPubSub(ct, l)
        ct.sim_local.sensor_callback = None
        ct.srvp.ode_pause()
    l.dpl.EndEpisode()

    print 'Copying', PycToPy(__file__), 'to', PycToPy(l.logdir+os.path.basename(__file__))
    CopyFile(PycToPy(__file__), PycToPy(l.logdir+os.path.basename(__file__)))

    return l
