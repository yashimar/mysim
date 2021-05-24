from ..util import CreatePredictionLog
from tsim.dpl_cmn import *
from core_tool import *
SmartImportReload('tsim.dpl_cmn')


def Delta1(dim, s):
    assert(abs(s-int(s)) < 1.0e-6)
    p = [0.0]*dim
    p[int(s)] = 1.0
    return p


def Rflowedout():
    modeldir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'\
        + 'reward_model'+"/"
    FRwd = TNNRegression()
    # prefix = modeldir+'p1_model/FRwdDefault'
    prefix =
    FRwd.Load(LoadYAML(prefix+'.yaml'), prefix)
    FRwd.Init()

    return FRwd


def Domain():  # SpaceDefs and Models (reward function) will be modified by curriculum. (For example, 'action' -> 'state', Rdamount -> Rdspill, and so on.)
    domain = TGraphDynDomain()
    SP = TCompSpaceDef
    domain.SpaceDefs = {
        'skill': SP('select', num=2),  # Skill selection
        # 'ps_rcv': SP('state', 12),  # 4 edge point positions (x,y,z)*4 of receiver
        'gh_ratio': SP('action', 1, min=[0.0], max=[1.0]),  # Gripper height (ratio)
        'gh_abs': SP('state', 1),  # Gripper height (absolute value)
        # 'p_pour_trg0': SP('state', 1, min=[0.1], max=[0.7]),  # Target pouring axis position of preparation before pouring (x,z)
        # NOTE: we stopped to plan p_pour_trg0
        'p_pour_trg0': SP('action', 1, min=[0.1], max=[0.7]),  # Target pouring axis position (x,z)
        'dtheta1': SP('action', 1, min=[0.01], max=[0.02]),  # Pouring skill parameter for all skills
        'dtheta2': SP('action', 1, min=[0.002], max=[0.02]),  # Pouring skill parameter for 'tip'
        'shake_spd': SP('action', 1, min=[0.5], max=[1.2]),  # Pouring skill parameter for 'shake'
        'shake_range': SP('action', 1, min=[0.05], max=[0.12]),
        'shake_angle': SP('action', 1, min=[-0.5*math.pi], max=[0.5*math.pi]),
        # 'shake_axis2': SP('action',2,min=[0.05,-0.5*math.pi],max=[0.1,0.5*math.pi]),
        'p_pour_z': SP('state', 1),  # Pouring axis position (z)
        # 'lp_pour': SP('state', 3),  # Pouring axis position (x,y,z) in receiver frame
        # 'dps_rcv': SP('state', 12),  # Displacement of ps_rcv from previous time
        # 'v_rcv': SP('state', 1),  # Velocity norm of receiver
        # 'p_flow': SP('state',2),  #Flow position (x,y)
        # 'lp_flow': SP('state', 2),  # Flow position (x,y) in receiver frame
        'lpp_flow': SP('state', 2),  # Flow position (x,y) relative to previous (before flowctrl) p_pour
        'flow_var': SP('state', 1),  # Variance of flow
        # 'a_pour': SP('state', 1),  # Amount poured in receiver
        # 'a_spill2': SP('state', 1),  # Amount spilled out
        'a_total':  SP('state', 1),  # Total amount moved from source
        'a_trg': SP('state', 1),  # Target amount
        # 'da_pour': SP('state', 1),  # Amount poured in receiver (displacement)
        # 'da_spill2': SP('state', 1),  # Amount spilled out (displacement)
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
        'Fmvtoinist': [  # Move to receiver
            ['gh_abs', 'p_pour_trg0'],
            ['p_pour_z'], None],
        # 'Fmvtorcv_rcvmv': [  # Move to receiver: receiver movement
        #     ['ps_rcv', 'gh_abs', 'p_pour', 'p_pour_trg0'],
        #     ['dps_rcv', 'v_rcv'], None],
        # 'Fmvtopour2': [  # Move to pouring point
        ##     ['ps_rcv', 'gh_abs', 'p_pour', 'p_pour_trg'],
        # ['lp_pour'], None],
        # 'Fmvtopour2': [  # Move to pouring point
        #     ['gh_abs', 'p_pour_trg'],
        #     ['lp_pour'], None],
        'Ftip': [  # Flow control with tip.
            ['gh_abs', 'p_pour_z',  # Removed 'p_pour_trg0','p_pour_trg'
             'da_trg', 'size_srcmouth', 'material2',
             'dtheta1', 'dtheta2'],
            ['da_total', 'lpp_flow', 'flow_var'], None],
        'Fshake': [  # Flow control with shake.
            ['gh_abs', 'p_pour_z',  # Removed 'p_pour_trg0','p_pour_trg'
             'da_trg', 'size_srcmouth', 'material2',
             'dtheta1', 'shake_spd', 'shake_range', 'shake_angle'],
            ['da_total', 'lpp_flow', 'flow_var'], None],
        # 'Famount4': [  # Amount model common for tip and shake.
        #     ['lp_pour',  # Removed 'gh_abs','p_pour_trg0','p_pour_trg'
        #      'da_trg', 'material2',  # Removed 'size_srcmouth'
        #      'da_total', 'lp_flow', 'flow_var'],
        #     ['da_pour', 'da_spill2'], None],
        # "Rdamount": [['da_pour', "da_spill2"], [REWARD_KEY], Rdamount()],
        "Rflowedout": [["da_trg", "da_total"], [REWARD_KEY], Rflowedout()],
        'P1': [[], [PROB_KEY], TLocalLinear(0, 1, lambda x:[1.0], lambda x:[0.0])],
        'P2':  [[], [PROB_KEY], TLocalLinear(0, 2, lambda x:[1.0]*2, lambda x:[0.0]*2)],
        'Pskill': [['skill'], [PROB_KEY], TLocalLinear(0, 2, lambda s:Delta1(2, s[0]), lambda s:[0.0]*2)],
    }
    domain.Graph = {
        'n0': TDynNode(None, 'P1', ('Fgrasp', 'n1')),
        'n1': TDynNode('n0', 'P1', ('Fmvtoinist', 'n2')),
        'n2': TDynNode('n1', 'Pskill', ('Ftip', 'n3ti'), ('Fshake', 'n3sa')),
        # Tipping:
        'n3ti': TDynNode('n2', 'P1', ('Rflowedout', 'n3tir')),
        'n3tir': TDynNode('n3ti'),
        # Shaking-A:
        'n3sa': TDynNode('n2', 'P1', ('Rflowedout', 'n3sar')),
        'n3sar': TDynNode('n3sa'),
    }

    return domain


def ConfigCallback(ct, l, sim):  # This will be modified by task's setup. (For example, l.custom_mtr -> "natto", l.custom_smsz -> 0.055, and so on.)
    m_setup = ct.Load('mysim.curriculum.tasks_domain.flowing_out.setup')
    l.amount_trg = 0.3
    # l.spilled_stop = 10
    l.config.RcvPos = [10, l.config.RcvPos[1], l.config.RcvPos[2]]
    # l.config.RcvPos= [0.8+0.6*(random.random()-0.5), l.config.RcvPos[1], l.config.RcvPos[2]]
    # CPrint(3, 'l.config.RcvPos=', l.config.RcvPos)
    for key, value in l.opt_conf['config'].iteritems():
        setattr(l.config, key, value)

    if l.rcv_size == 'static':
        l.config.RcvSize = [0.3, 0.4, 0.2]
    elif l.rcv_size == 'random':
        rsx = Rand(0.25, 0.5)
        rsy = Rand(0.1, 0.2)/rsx
        rsz = Rand(0.2, 0.5)
        l.config.RcvSize = [rsx, rsy, rsz]

    if l.mtr_smsz == 'fixed':
        m_setup.SetMaterial(l, preset='bounce')
        l.config.SrcSize2H = 0.03  # Mouth size of source container
    elif l.mtr_smsz == 'fxvs1':
        m_setup.SetMaterial(l, preset='ketchup')
        l.config.SrcSize2H = 0.08  # Mouth size of source container
    elif l.mtr_smsz == 'random':
        m_setup.SetMaterial(l, preset=('bounce', 'nobounce', 'natto', 'ketchup')[RandI(4)])
        l.config.SrcSize2H = Rand(0.03, 0.08)  # Mouth size of source container
    elif l.mtr_smsz == 'viscous':
        m_setup.SetMaterial(l, preset=('natto', 'ketchup')[RandI(2)])
        l.config.SrcSize2H = Rand(0.03, 0.08)  # Mouth size of source container
    elif l.mtr_smsz == 'curriculum_test':
        m_setup.SetMaterial(l, preset=('nobounce', 'ketchup')[RandI(2)])
        l.config.SrcSize2H = Rand(0.03, 0.08)
    elif l.mtr_smsz == "middle_ease_of_flow":
        mtr, smsz = (("nobounce", 0.055), ("nobounce", 0.08), ("ketchup", 0.03), ("ketchup", 0.055))[RandI(4)]
        m_setup.SetMaterial(l, preset=mtr)
        l.config.SrcSize2H = smsz
    elif l.mtr_smsz == "custom":
        if l.custom_mtr == "random":
            l.latest_mtr = ('bounce', 'nobounce', 'natto', 'ketchup')[RandI(4)]
            m_setup.SetMaterial(l, preset=l.latest_mtr)
        else:
            l.latest_mtr = l.custom_mtr
            m_setup.SetMaterial(l, preset=l.custom_mtr)
        if l.custom_smsz == "random":
            l.config.SrcSize2H = Rand(0.03, 0.08)
        else:
            l.config.SrcSize2H = l.custom_smsz
        l.latest_smsz = l.config.SrcSize2H
    elif l.mtr_smsz == "latest_mtr_smsz":
        m_setup.SetMaterial(l, preset=l.latest_mtr)
        l.config.SrcSize2H = Rand(max(0.03, l.latest_smsz-l.delta_smsz),
                                  min(0.08, l.latest_smsz+l.delta_smsz))
    elif l.mtr_smsz == "early_natto":
        m_setup.SetMaterial(l, preset='natto')
        l.config.SrcSize2H = l.custom_smsz
    CPrint(3, 'l.config.ViscosityParam1=', l.config.ViscosityParam1)
    CPrint(3, 'l.config.SrcSize2H=', l.config.SrcSize2H)


def Execute(ct, l):
    l.node_best_tree = []
    l.pred_true_log = []
    l.user_viz = []  # Use in dpl_cmn

    l.dpl.NewEpisode()
    try:
        ct.Run('mysim.curriculum.tasks_domain.flowing_out.setup', l)
        sim = ct.sim

        actions = {
            'grab': lambda a: ct.Run('mysim.act.grab_sv', a),
            'move_to_init_state': lambda a: ct.Run('mysim.act.move_to_init_state', a),  # For initial pouring pose.
            'tip': lambda a: ct.Run('mysim.act.tip', a),
            'shake': lambda a: ct.Run('mysim.act.shake', a),
        }

        obs_keys0 = (
            # 'ps_rcv',
            # 'p_pour',
            "p_pour_z",
            # 'lp_pour',
            'a_trg',
            'size_srcmouth',
            'material2'
        )
        obs_keys_after_grab = obs_keys0+('gh_abs',)
        obs_keys_before_flow = obs_keys_after_grab + \
            (
                # 'a_pour',
                # 'a_spill2',
                'a_total'
            )
        obs_keys_after_flow = obs_keys_before_flow + \
            (
                # 'lp_flow',
                'lpp_flow',
                'flow_var',
                # 'da_pour',
                # 'da_spill2',
                'da_total'
            )

        l.xs = TContainer()  # l.xs.NODE= XSSA
        l.idb = TContainer()  # l.idb.NODE= index in DB

        ############################################################################
        # n0: Initialize skill params, Plan, Execute grab
        ############################################################################
        CPrint(2, 'Node:', 'n0')
        l.xs.n0 = ObserveXSSA(l, None, obs_keys0+('da_trg',))
        # Heuristic init guess
        # # pc_rcv = np.array(l.xs.n0['ps_rcv'].X).reshape(4, 3).mean(axis=0)  # Center of ps_rcv
        # # l.xs.n0['p_pour_trg0'] = SSA(Vec([-0.3, 0.35])+Vec([pc_rcv[0], pc_rcv[2]]))
        # # l.xs.n0['p_pour_trg0'] = SSA(Vec([0., Rand(0.1, 0.7)]))
        l.xs.n0['gh_ratio'] = SSA([0.5])
        # # l.xs.n0['p_pour_trg'] = SSA(Vec([Rand(0.2, 1.2), Rand(0.1, 0.7)]))
        # l.xs.n0['dtheta1'] = SSA([0.014])
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
        # n1: Update Fgrasp, Plan (option), Execute move_to_init_state
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
        # p_pour_trg0 = ToList(l.xs.n1['p_pour_trg0'].X)
        # p_pour_trg = ToList(l.xs.n1['p_pour_trg'].X)
        p_pour_trg0 = Vec([0., l.xs.n1['p_pour_trg0'].X])
        actions['move_to_init_state']({'p_pour_trg0': p_pour_trg0})
        VizPP(l, [0.0, 0.0, p_pour_trg0[1]], [0., 1., 0.])
        # VizPP(l, [p_pour_trg[0], 0.0, p_pour_trg[1]], [0.5, 0., 1.])

        ############################################################################
        # n2: Update Fmvtoinist (only first time), Plan (option)
        ############################################################################
        CPrint(2, 'Node:', 'n2')
        l.xs.n2 = CopyXSSA(l.xs.prev)
        CreatePredictionLog(l, "Fmvtoinist", l.xs.prev, l.xs.n2)
        l.dpl.MM.Models['Fmvtoinist'][2].Options.update(l.nn_options)
        l.dpl.MM.Update('Fmvtoinist', l.xs.prev, l.xs.n2, not_learn=l.not_learn)

        repeated = False  # For try-and-error learning
        while True:  # Try-and-error starts from here.
            # CPrint(2, 'Node:', 'n2a')
            # l.xs.n2a = CopyXSSA(l.xs.prev)
            if repeated:
                # Delete actions and selections (e.g. skill) to plan again from initial guess.
                for key in l.xs.n2.keys():
                    if l.dpl.d.SpaceDefs[key].Type in ('action', 'select'):
                        del l.xs.n2[key]
            InsertDict(l.xs.n2, ObserveXSSA(l, l.xs.prev, obs_keys_after_grab+('da_trg',)))
            if "n2" in l.planning_node:
                res = l.dpl.Plan('n2', l.xs.n2, l.interactive)
                l.node_best_tree.append(res.PTree)
            l.idb.n2 = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n2', xs=l.xs.n2)
            l.xs.prev = l.xs.n2
            l.idb.prev = l.idb.n2

            ############################################################################
            # n2 (pouring): Execute pouring skill (Selection is already planned before node)
            ############################################################################
            idx = int(l.xs.n2c['skill'].X[0])
            selected_skill = ('tip', 'shake')[idx]

            if selected_skill == 'tip':
                dtheta1 = l.xs.n2c['dtheta1'].X[0, 0]
                dtheta2 = l.xs.n2c['dtheta2'].X[0, 0]
                actions['tip']({'dtheta1': dtheta1, 'dtheta2': dtheta2})

                ############################################################################
                # n3ti: Update Ftip
                ############################################################################
                CPrint(2, 'Node:', 'n3ti')
                l.xs.n3ti = CopyXSSA(l.xs.prev)
                InsertDict(l.xs.n3ti, ObserveXSSA(l, l.xs.prev, obs_keys_after_flow))
                CreatePredictionLog(l, "Ftip", l.xs.prev, l.xs.n3ti)
                l.dpl.MM.Models['Ftip'][2].Options.update(l.nn_options)
                l.dpl.MM.Update('Ftip', l.xs.prev, l.xs.n3ti, not_learn=l.not_learn)
                if "n3ti" in l.planning_node:
                    res = l.dpl.Plan('n3ti', l.xs.n3ti, l.interactive)
                    l.node_best_tree.append(res.PTree)
                l.idb.n3ti = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n3ti', xs=l.xs.n3ti)
                l.xs.prev = l.xs.n3ti
                l.idb.prev = l.idb.n3ti

                ############################################################################
                # n3tir: Calculate Rdamount
                ############################################################################
                CPrint(2, 'Node:', 'n3tir')
                l.xs.n3tir = l.dpl.Forward('Rflowedout', l.xs.prev)
                l.idb.n3tir = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n3tir', xs=l.xs.n3tir)

            elif selected_skill == 'shake':
                dtheta1 = l.xs.n2c['dtheta1'].X[0, 0]
                shake_spd = l.xs.n2c['shake_spd'].X[0, 0]
                # shake_axis2 = ToList(l.xs.n2c['shake_axis2'].X)
                shake_axis2 = ToList([l.xs.n2c['shake_range'].X.item(), l.xs.n2c['shake_angle'].X.item()])
                actions['shake']({'dtheta1': dtheta1, 'shake_spd': shake_spd, 'shake_axis2': shake_axis2})

                ############################################################################
                # n3sa: Update Fshake
                ############################################################################
                CPrint(2, 'Node:', 'n3sa')
                l.xs.n3sa = CopyXSSA(l.xs.prev)
                InsertDict(l.xs.n3sa, ObserveXSSA(l, l.xs.prev, obs_keys_after_flow))
                CreatePredictionLog(l, "Fshake", l.xs.prev, l.xs.n3sa)
                l.dpl.MM.Models['Fshake'][2].Options.update(l.nn_options)
                l.dpl.MM.Update('Fshake', l.xs.prev, l.xs.n3sa, not_learn=l.not_learn)
                if "n3sa" in l.planning_node:
                    res = l.dpl.Plan('n3sa', l.xs.n3sa, l.interactive)
                    l.node_best_tree.append(res.PTree)
                l.idb.n3sa = l.dpl.DB.AddToSeq(parent=l.idb.prev, name='n3sa', xs=l.xs.n3sa)
                l.xs.prev = l.xs.n3sa
                l.idb.prev = l.idb.n3sa

                ############################################################################
                # n3sar: Caluculate Rdamount
                ############################################################################
                CPrint(2, 'Node:', 'n3sar')
                l.xs.n3sar = l.dpl.Forward('Rdamount', l.xs.n3sar)
                l.idb.n3sar = l.dpl.DB.AddToSeq(parent=l.idb.n3sar, name='n3sar', xs=l.xs.n3sar)

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
