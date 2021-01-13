#!/usr/bin/python
from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from mysim.dpl_cmn import *
import joblib
import GPyOpt
def Help():
  return '''Dynamic Planning/Learning for grasping and pouring in ODE simulation
    using DPL version 4 (DNN, Bifurcation).
    Based on tsim.dplD14, modified for the new action system (cf. tsim2.test_dpl1a).
    The behavior is the same as tsim.dplD14.
    We share the same model Famount_* in different skills.
  Usage: tsim2.dplD20'''

def Delta1(dim,s):
  assert(abs(s-int(s))<1.0e-6)
  p= [0.0]*dim
  p[int(s)]= 1.0
  return p

def Execute(ct,l):
  ct.Run('mysim.setup.setup_sv', l)
  sim= ct.sim
  #l= ct.sim_local

  actions={
    'grab'         : lambda a: ct.Run('mysim.act.grab_sv', a),
    'move_to_rcv'  : lambda a: ct.Run('mysim.act.move_to_rcv_sv', a),
    'move_to_pour' : lambda a: ct.Run('mysim.act.move_to_pour_sv', a),
    'std_pour'     : lambda a: ct.Run('mysim.act.std_pour_sv', a),
    'shake_A'      : lambda a: ct.Run('mysim.act.shake_A_5s_sv', a),
    }

  #NOTE: Do not include 'da_trg' and "a_src" in obs_keys0 since 'da_trg' and "a_src" should be kept during some node transitions.
  obs_keys0= ('ps_rcv','p_pour','lp_pour','a_trg','size_srcmouth','material2')
  obs_keys_after_grab= obs_keys0+('gh_abs',)
  obs_keys_before_flow= obs_keys_after_grab+('a_pour','a_spill2','a_total')
  obs_keys_after_flow= obs_keys_before_flow+('lp_flow','flow_var','da_pour','da_spill2','da_total')

  l.xs= TContainer()  #l.xs.NODE= XSSA
  l.idb= TContainer()  #l.idb.NODE= index in DB

  with sim.TPause(ct):
    CPrint(2,'Node:','n0')
    l.xs.n0= ObserveXSSA(l,None,obs_keys0+('da_trg',"a_src","a_spill2",))

    pc_rcv= np.array(l.xs.n0['ps_rcv'].X).reshape(4,3).mean(axis=0)  #Center of ps_rcv
    l.xs.n0['gh_ratio']= SSA([0.5])
    l.xs.n0['p_pour_trg0']= SSA(Vec([-0.3,0.35])+Vec([pc_rcv[0],pc_rcv[2]]))  #A bit above of p_pour_trg
    l.xs.n0['dtheta1']= SSA([0.014])
    l.xs.n0['dtheta2']= SSA([0.002])
    l.xs.n0['shake_spd']= SSA([0.8])
    # l.xs.n0['shake_axis2']= SSA([0.08,0.0])
    
    # planed result into l.xs.n0
    res = l.dpl.Plan('n0', l.xs.n0, l.interactive)
    l.node_best_tree.append(res.PTree)

    l.idb.n0= l.dpl.DB.AddToSeq(parent=None,name='n0',xs=l.xs.n0)
    l.xs.prev= l.xs.n0
    l.idb.prev= l.idb.n0
  
  gh_ratio= ToList(l.xs.n0['gh_ratio'].X)[0]
  actions['grab']({'gh_ratio':gh_ratio})

  with sim.TPause(ct):  #Pause during plan/learn
    #Plan l.p_pour_trg0, l.theta_init
    CPrint(2,'Node:','n1')
    l.xs.n1= CopyXSSA(l.xs.prev)
    InsertDict(l.xs.n1, ObserveXSSA(l,l.xs.prev,obs_keys_after_grab))
    # l.dpl.MM.Models['Fgrasp'][2].Options.update(l.nn_options)
    # l.dpl.MM.Update('Fgrasp',l.xs.prev,l.xs.n1, not_learn=l.not_learn)
    #res= l.dpl.Plan('n1', l.xs.n1)
    l.idb.n1= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n1',xs=l.xs.n1)
    l.xs.prev= l.xs.n1
    l.idb.prev= l.idb.n1

  p_pour_trg0= ToList(l.xs.n1['p_pour_trg0'].X)
  p_pour_trg= ToList(l.xs.n1['p_pour_trg'].X) #l.xs.n0['p_pour_trg'].X?
  actions['move_to_rcv']({'p_pour_trg0':p_pour_trg0})
  VizPP(l,[p_pour_trg0[0],0.0,p_pour_trg0[1]],[0.,1.,0.])
  VizPP(l,[p_pour_trg[0],0.0,p_pour_trg[1]],[0.5,0.,1.])

  with sim.TPause(ct):  #Pause during plan/learn
    #Branch-1: reward
    CPrint(2,'Node:','n1rcvmv')
    l.xs.n1rcvmv= CopyXSSA(l.xs.prev)
    InsertDict(l.xs.n1rcvmv, ObserveXSSA(l,l.xs.prev,('dps_rcv','v_rcv')))
    # l.dpl.MM.Models['Fmvtorcv_rcvmv'][2].Options.update(l.nn_options)
    # l.dpl.MM.Update('Fmvtorcv_rcvmv',l.xs.prev,l.xs.n1rcvmv, not_learn=l.not_learn)
    #res= l.dpl.Plan('n1rcvmv', l.xs.n1rcvmv)
    l.idb.n1rcvmv= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n1rcvmv',xs=l.xs.n1rcvmv)

    # CPrint(2,'Node:','n1rcvmvr')
    # #Since we have 'Rrcvmv', we just use it to get the next XSSA
    # l.xs.n1rcvmvr= l.dpl.Forward('Rrcvmv',l.xs.n1rcvmv)
    # l.idb.n1rcvmvr= l.dpl.DB.AddToSeq(parent=l.idb.n1rcvmv,name='n1rcvmvr',xs=l.xs.n1rcvmvr)

    #Branch-2: main procedure
    CPrint(2,'Node:','n2a','(update)')
    l.xs.n2a= CopyXSSA(l.xs.prev)
    InsertDict(l.xs.n2a, ObserveXSSA(l,l.xs.prev,obs_keys_after_grab))
    # l.dpl.MM.Models['Fmvtorcv'][2].Options.update(l.nn_options)
    # l.dpl.MM.Update('Fmvtorcv',l.xs.prev,l.xs.n2a, not_learn=l.not_learn)



  repeated= False  #For try-and-error learning
  while True:  #Try-and-error starts from here.
    #Three cases of parent of l.idb.n2a: l.idb.n1, l.idb.n4ti, l.idb.n4sa

    with sim.TPause(ct):  #Pause during plan/learn
      #Plan l.p_pour_trg
      CPrint(2,'Node:','n2a','(plan)')
      l.xs.n2a= CopyXSSA(l.xs.prev)
      if repeated:
        #Delete actions and selections (e.g. skill) to plan again from initial guess.
        for key in l.xs.n2a.keys():
          if l.dpl.d.SpaceDefs[key].Type in ('action','select'):
            del l.xs.n2a[key]
      InsertDict(l.xs.n2a, ObserveXSSA(l,l.xs.prev,obs_keys_after_grab+('da_trg',"a_src","a_spill2",)))
      #TEST: Heuristic init guess
      #l.xs.n2a['skill']= SSA([1])
      if "n2a" in l.planning_node and repeated:
        l.dpl.d.Models['Rdamount']= [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
                                      TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2)]
        res= l.dpl.Plan('n2a', l.xs.n2a, l.interactive)
        l.node_best_tree.append(res.PTree)
      # CPrint(2,"max return estimation:",l.dpl.Value(res.PTree))
      # CPrint(2,"start node XS:",res.XS)
      l.idb.n2a= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n2a',xs=l.xs.n2a)
      l.xs.prev= l.xs.n2a
      l.idb.prev= l.idb.n2a

    p_pour_trg= ToList(l.xs.n2a['p_pour_trg'].X)
    actions['move_to_pour']({'p_pour_trg':p_pour_trg})
    l.user_viz.pop()
    VizPP(l,[p_pour_trg[0],0.0,p_pour_trg[1]],[1.,0.,1.])


    with sim.TPause(ct):  #Pause during plan/learn
      CPrint(2,'Node:','n2b')
      l.xs.n2b= CopyXSSA(l.xs.prev)
      InsertDict(l.xs.n2b, ObserveXSSA(l,l.xs.prev,obs_keys_after_grab))
      l.dpl.MM.Models['Fmvtopour2'][2].Options.update(l.nn_options)
      l.dpl.MM.Update('Fmvtopour2',l.xs.prev,l.xs.n2b, not_learn=l.not_learn)
      #res= l.dpl.Plan('n2b', l.xs.n2b)
      l.idb.n2b= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n2b',xs=l.xs.n2b)
      l.xs.prev= l.xs.n2b
      l.idb.prev= l.idb.n2b

    #Branch-1: main procedure
    #Just go to 'n2c'

    #Branch-2: reward
    # CPrint(2,'Node:','n2br')
    # #Since we have 'Rmvtopour', we just use it to get the next XSSA
    # l.xs.n2br= l.dpl.Forward('Rmvtopour',l.xs.prev)
    # l.idb.n2br= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n2br',xs=l.xs.n2br)

    with sim.TPause(ct):  #Pause during plan/learn
      #Plan l.selected_skill from ('std_pour','shake_A','shake_B')
      CPrint(2,'Node:','n2c')
      l.xs.n2c= CopyXSSA(l.xs.prev)
      InsertDict(l.xs.n2c, ObserveXSSA(l,l.xs.prev,obs_keys_before_flow+("a_spill2",)))
      #l.dpl.MM.Update('Fnone',l.xs.prev,l.xs.n2c, not_learn=l.not_learn)
      #res= l.dpl.Plan('n2c', l.xs.n2c)
      l.idb.n2c= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n2c',xs=l.xs.n2c)
      l.xs.prev= l.xs.n2c
      l.idb.prev= l.idb.n2c

    if l.pour_skill=="std_pour": idx = 0
    elif l.pour_skill=="shake_A": idx = 1
    elif l.pour_skill=="choose": idx = int(l.xs.n2c['skill'].X[0])
    
    selected_skill= ('std_pour','shake_A')[idx]

    if selected_skill=='std_pour':
      dtheta1= l.xs.n2c['dtheta1'].X[0,0]
      dtheta2= l.xs.n2c['dtheta2'].X[0,0]
      actions['std_pour']({'dtheta1':dtheta1, 'dtheta2':dtheta2})

      with sim.TPause(ct):  #Pause during plan/learn
        CPrint(2,'Node:','n3ti')
        l.xs.n3ti= CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n3ti, ObserveXSSA(l,l.xs.prev,obs_keys_after_flow))
        # l.dpl.MM.Models['Fflowc_tip10'][2].Options.update(l.nn_options)
        # l.dpl.MM.Update('Fflowc_tip10',l.xs.prev,l.xs.n3ti, not_learn=l.not_learn)
        #res= l.dpl.Plan('n3ti', l.xs.n3ti)
        l.idb.n3ti= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n3ti',xs=l.xs.n3ti)
        l.xs.prev= l.xs.n3ti
        l.idb.prev= l.idb.n3ti

        CPrint(2,'Node:','n4ti')
        l.xs.n4ti= CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n4ti, ObserveXSSA(l,l.xs.prev,()))  #Observation is omitted since there is no change
        #WARNING:NOTE: Famount4 uses 'lp_pour', "a_spill2" as input, so here we use a trick:
        xs_in= CopyXSSA(l.xs.prev)
        xs_in['lp_pour']= l.xs.n2c['lp_pour']
        xs_in['a_spill2']= l.xs.n2c['a_spill2']
        #l.dpl.MM.Update('Famount4',l.xs.prev,l.xs.n4ti, not_learn=l.not_learn)
        l.dpl.MM.Models['Fflowc_tip10'][2].Options.update(l.nn_options)
        l.dpl.MM.Update('Fflowc_tip10',xs_in,l.xs.n4ti, not_learn=l.not_learn)
        #res= l.dpl.Plan('n4ti', l.xs.n4ti)
        l.idb.n4ti= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n4ti',xs=l.xs.n4ti)
        l.xs.prev= l.xs.n4ti
        l.idb.prev= l.idb.n4ti

        CPrint(2,'Node:','n4tir')
        l.xs.n4tir= l.dpl.Forward('Rdamount',l.xs.prev)
        l.idb.n4tir= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n4tir',xs=l.xs.n4tir)
        l.true_return.append(l.xs.n4tir[".r"].X.item())

    elif selected_skill=='shake_A':
      dtheta1= l.xs.n2c['dtheta1'].X[0,0]
      shake_spd= l.xs.n2c['shake_spd'].X[0,0]
      shake_axis2= ToList(l.xs.n2c['shake_axis2'].X)
      actions['shake_A']({'dtheta1':dtheta1, 'shake_spd':shake_spd, 'shake_axis2':shake_axis2})

      with sim.TPause(ct):  #Pause during plan/learn
        CPrint(2,'Node:','n3sa')
        l.xs.n3sa= CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n3sa, ObserveXSSA(l,l.xs.prev,obs_keys_after_flow))
        # l.dpl.MM.Models['Fflowc_shakeA10'][2].Options.update(l.nn_options)
        # l.dpl.MM.Update('Fflowc_shakeA10',l.xs.prev,l.xs.n3sa, not_learn=l.not_learn)
        #res= l.dpl.Plan('n3sa', l.xs.n3sa)
        l.idb.n3sa= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n3sa',xs=l.xs.n3sa)
        l.xs.prev= l.xs.n3sa
        l.idb.prev= l.idb.n3sa

        CPrint(2,'Node:','n4sa')
        l.xs.n4sa= CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n4sa, ObserveXSSA(l,l.xs.prev,()))  #Observation is omitted since there is no change
        #WARNING:NOTE: Famount4 uses 'lp_pour', "a_spill2" as input, so here we use a trick:
        xs_in= CopyXSSA(l.xs.prev)
        xs_in['lp_pour']= l.xs.n2c['lp_pour']
        xs_in['a_spill2']= l.xs.n2c['a_spill2']
        #l.dpl.MM.Update('Famount4',l.xs.prev,l.xs.n4sa, not_learn=l.not_learn)
        l.dpl.MM.Models['Fflowc_shakeA10'][2].Options.update(l.nn_options)
        l.dpl.MM.Update('Fflowc_shakeA10',xs_in,l.xs.n4sa, not_learn=l.not_learn)
        #res= l.dpl.Plan('n4sa', l.xs.n4sa)
        l.idb.n4sa= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n4sa',xs=l.xs.n4sa)
        l.xs.prev= l.xs.n4sa
        l.idb.prev= l.idb.n4sa

        CPrint(2,'Node:','n4sar')
        l.xs.n4sar= l.dpl.Forward('Rdamount',l.xs.prev)
        l.idb.n4sar= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n4sar',xs=l.xs.n4sar)
        l.true_return.append(l.xs.n4sar[".r"].X.item())

    if "n2a" in l.planning_node:
      # Conditions to break the try-and-error loop
      if l.IsPoured():
        break
      if l.IsTimeout() or l.IsEmpty():  # or l.IsSpilled()
        break
      if not IsSuccess(l.exec_status):
        break
      repeated= True
    else:
      break



def Run(ct,*args):
  l = args[0]

  l.interactive= l.opt_conf['interactive']
  l.num_episodes= l.opt_conf['num_episodes']
  l.max_priority_sampling = l.opt_conf["max_priority_sampling"]
  # l.sampling_mode = l.opt_conf["sampling_mode"]
  l.return_epsiron = l.opt_conf["return_epsiron"]
  l.num_log_interval= l.opt_conf['num_log_interval']
  l.planning_node = l.opt_conf["planning_node"]
  l.rcv_size= l.opt_conf['rcv_size']
  l.mtr_smsz= l.opt_conf['mtr_smsz']
  l.rwd_schedule= l.opt_conf['rwd_schedule']
  l.mtr_schedule = l.opt_conf['mtr_schedule']

  l.not_learn= l.opt_conf['not_learn']
  l.config_log = []
  
  l.org_not_learn = l.not_learn
  l.org_planning_node = l.planning_node
  l.org_mtr_smsz = l.mtr_smsz

  #Setup dynamic planner/learner
  domain= TGraphDynDomain()
  SP= TCompSpaceDef
  domain.SpaceDefs={
    'skill': SP('select',num=2),  #Skill selection
    'ps_rcv': SP('state',12),  #4 edge point positions (x,y,z)*4 of receiver
    'gh_ratio': SP('state',1,min=[0.0],max=[1.0]),  #Gripper height (ratio)
    'gh_abs': SP('state',1),  #Gripper height (absolute value)
    'p_pour_trg0': SP('state',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position of preparation before pouring (x,z)
      #NOTE: we stopped to plan p_pour_trg0
    'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position (x,z)
    'dtheta1': SP('state',1,min=[0.01],max=[0.02]),  #Pouring skill parameter for all skills
    'dtheta2': SP('state',1,min=[0.002],max=[0.005]),  #Pouring skill parameter for 'std_pour'
    #'dtheta1': SP('state',1),  #Pouring skill parameter for all skills
    #'dtheta2': SP('state',1),  #Pouring skill parameter for 'std_pour'
    'shake_spd': SP('state',1,min=[0.7],max=[0.9]),  #Pouring skill parameter for 'shake_A'
    #'shake_spd': SP('state',1),  #Pouring skill parameter for 'shake_A'
    #'shake_axis': SP('action',2,min=[0.0,0.0],max=[0.1,0.1]),  #Pouring skill parameter for 'shake_A'
    'shake_axis2': SP('action',2,min=[0.01,-0.5*math.pi],max=[0.1,0.5*math.pi]),  #Pouring skill parameter for 'shake_A'
    #'shake_axis2': SP('state',2),  #Pouring skill parameter for 'shake_A'
    'shake_spd_B': SP('state',1,min=[2.0],max=[8.0]),  #Pouring skill parameter for 'shake_B'
    "shake_range" : SP('state',1,min=[0.02],max=[0.06]),  #Pouring skill parameter for 'shake_B'
    'p_pour': SP('state',3),  #Pouring axis position (x,y,z)
    'lp_pour': SP('state',3),  #Pouring axis position (x,y,z) in receiver frame
    'dps_rcv': SP('state',12),  #Displacement of ps_rcv from previous time
    'v_rcv': SP('state',1),  #Velocity norm of receiver
    #'p_flow': SP('state',2),  #Flow position (x,y)
    'lp_flow': SP('state',2),  #Flow position (x,y) in receiver frame
    'flow_var': SP('state',1),  #Variance of flow
    'a_pour': SP('state',1),  #Amount poured in receiver
    'a_spill2': SP('state',1),  #Amount spilled out
    'a_total':  SP('state',1),  #Total amount moved from source
    "a_src": SP('state',1),
    'a_trg': SP('state',1),  #Target amount
    'da_pour': SP('state',1),  #Amount poured in receiver (displacement)
    'da_spill2': SP('state',1),  #Amount spilled out (displacement)
    'da_total':  SP('state',1),  #Total amount moved from source (displacement)
    'da_trg': SP('state',1),  #Target amount (displacement)
    'size_srcmouth': SP('state',1),  #Size of mouth of the source container
    'material2': SP('state',4),  #Material property (e.g. viscosity)
    REWARD_KEY:  SP('state',1),
    }
  domain.Models={
    #key:[In,Out,F],
    'Fnone': [[],[], None],
    # 'Fgrasp': [['gh_ratio'],['gh_abs'],None],  #Grasping. NOTE: removed ps_rcv
    # 'Fmvtorcv': [  #Move to receiver
    #   ['ps_rcv','gh_abs','p_pour','p_pour_trg0'],
    #   ['ps_rcv','p_pour'],None],
    # 'Fmvtorcv_rcvmv': [  #Move to receiver: receiver movement
    #   ['ps_rcv','gh_abs','p_pour','p_pour_trg0'],
    #   ['dps_rcv','v_rcv'],None],
    # 'Fmvtopour2': [  #Move to pouring point
    #   ['ps_rcv','gh_abs','p_pour','p_pour_trg'],
    #   ['lp_pour'],None],
    'Fmvtopour2': [  #Move to pouring point
      ['p_pour_trg'],
      ['lp_pour'],None],
    'Fflowc_tip10': [  #Flow control with tipping.
      ['lp_pour','size_srcmouth',
        "da_trg","a_src","a_spill2"],
      ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    # 'Fflowc_shakeA10': [  #Flow control with shake_A.
    #   ['gh_abs','lp_pour',  #Removed 'p_pour_trg0','p_pour_trg'
    #    'da_trg','size_srcmouth','material2',
    #    'dtheta1','shake_spd','shake_axis2'],
      # ['da_total','lp_flow','flow_var'],None],  #Removed 'p_pour'
    'Fflowc_shakeA10': [  #Flow control with shake_A.
      ['lp_pour','size_srcmouth','shake_axis2',
        "da_trg","a_src","a_spill2"],
      ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    # 'Fflowc_shakeB10': [  #Flow control with shake_B.
    #   ['gh_abs','lp_pour',  #Removed 'p_pour_trg0','p_pour_trg'
    #    'da_trg','size_srcmouth','material2',
    #    'dtheta1','shake_spd_B','shake_range'],
    #   ['da_total','lp_flow','flow_var'],None],  #Removed 'p_pour'
    # 'Famount4': [  #Amount model common for tip and shake.
    #   ['lp_pour',  #Removed 'gh_abs','p_pour_trg0','p_pour_trg'
    #    'da_trg','material2',  #Removed 'size_srcmouth'
    #    'da_total','lp_flow','flow_var'],
    #   ['da_pour','da_spill2'],None],
    # 'Rrcvmv':  [['dps_rcv','v_rcv'],[REWARD_KEY],TLocalQuad(13,lambda y:-(np.dot(y[:12],y[:12]) + y[12]*y[12]))],
    # 'Rmvtopour':  [['p_pour_trg','p_pour'],[REWARD_KEY],TLocalQuad(5,lambda y:-0.1*((y[0]-y[2])**2+(y[1]-y[4])**2))],
    #'Ramount':  [['a_pour','a_trg','a_spill2'],[REWARD_KEY],TLocalQuad(3,lambda y:-100.0*(y[1]-y[0])*(y[1]-y[0]) - y[2]*y[2])],
    #'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
                  #TLocalQuad(3,lambda y:-100.0*(y[1]-y[0])*(y[1]-y[0]) - y[2]*y[2])],
    #'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
                  #TLocalQuad(3,lambda y:-100.0*(y[1]-y[0])*(y[1]-y[0]) - math.log(1.0+max(0.0,y[2])))],
    #'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
                  #TLocalQuad(3,lambda y:-100.0*(y[1]-y[0])*(y[1]-y[0]) - max(0.0,y[2])**2)],
    'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
                  TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2)],
    #'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
                  #TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 10.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2)],
    'P1': [[],[PROB_KEY], TLocalLinear(0,1,lambda x:[1.0],lambda x:[0.0])],
    'P2':  [[],[PROB_KEY], TLocalLinear(0,2,lambda x:[1.0]*2,lambda x:[0.0]*2)],
    'Pskill': [['skill'],[PROB_KEY], TLocalLinear(0,2,lambda s:Delta1(2,s[0]),lambda s:[0.0]*2)],
    }
  domain.Graph={
    'n0': TDynNode(None,'P1',('Fnone','n1')),
    'n1': TDynNode('n0','P1',('Fnone','n2a')),
    'n2a': TDynNode('n1','P1',('Fmvtopour2','n2b')),
    'n2b': TDynNode('n2a','P1',('Fnone','n2c')),
    # 'n2c': TDynNode('n2b','Pskill',('Fflowc_tip10','n3ti'),('Fflowc_shakeA10','n3sa')),
    # "n2c": None, 
    'n2c': TDynNode('n2b','Pskill',('Fnone','n3ti'),('Fnone','n3sa')),
    #Tipping:
    'n3ti': TDynNode('n2c','P1',('Fflowc_tip10','n4ti')),
    'n4ti': TDynNode('n3ti','P1',('Rdamount','n4tir')),
    'n4tir': TDynNode('n4ti'),
    #Shaking-A:
    'n3sa': TDynNode('n2c','P1',('Fflowc_shakeA10','n4sa')),
    'n4sa': TDynNode('n3sa','P1',('Rdamount','n4sar')),
    'n4sar': TDynNode('n4sa'),
    }
  # if l.pour_skill=="std_pour":
  #   domain.Graph.update({'n2c': TDynNode('n2b','P1',('Fnone','n3ti'))})
  # elif l.pour_skill=="shake_A":
  #   domain.Graph.update({'n2c': TDynNode('n2b','P1',('Fnone','n3sa'))})
  # elif l.pour_skill=="choose":
  #   domain.Graph.update({'n2c': TDynNode('n2b','Pskill',('Fnone','n3ti'),('Fnone','n3sa'))})
  
  #Learning scheduling
  def EpisodicCallback(l,count):
    Rdamount_default= [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
          TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2)]
    Rdamount_early_tip= [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
          TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2 - (200.0 if y[3]!=0 else 0.0))]
    Rdamount_early_shakeA= [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
          TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2 - (200.0 if y[3]!=1 else 0.0))]
    #'rwd_schedule': None,  #None, 'early_tip', 'early_shakeA'
    if l.rwd_schedule is None:
      #No reward scheduling
      pass
    elif l.rwd_schedule=='early_tip':
      #Reward scheduling (FOR EARLY TIPPING)
      if count<200:  l.dpl.d.Models['Rdamount']= Rdamount_early_tip
      else:         l.dpl.d.Models['Rdamount']= Rdamount_default
    elif l.rwd_schedule=='early_shakeA':
      #Reward scheduling (FOR EARLY SHAKING-A)
      if count<200:  l.dpl.d.Models['Rdamount']= Rdamount_early_shakeA
      else:         l.dpl.d.Models['Rdamount']= Rdamount_default
    elif l.rwd_schedule=='early_tip_and_shakeA':
      if count<100:
        if count%2==0:  l.dpl.d.Models['Rdamount']= Rdamount_early_tip
        else:           l.dpl.d.Models['Rdamount']= Rdamount_early_shakeA
      else:             l.dpl.d.Models['Rdamount']= Rdamount_default
    elif l.rwd_schedule=='only_tip': l.dpl.d.Models['Rdamount']= Rdamount_early_tip
    elif l.rwd_schedule=='only_shakeA': l.dpl.d.Models['Rdamount']= Rdamount_early_shakeA

    if l.mtr_schedule==None:
      pass
    elif l.mtr_schedule=="early_nobounce":
      if count<100:  l.mtr_smsz = "early_nobounce"
      else:         l.mtr_smsz = l.org_mtr_smsz
    elif l.mtr_schedule=="early_bounce":
      if count<100:  l.mtr_smsz = "early_bounce"
      else:         l.mtr_smsz = l.org_mtr_smsz
    elif l.mtr_schedule=="early_ketchup":
      if count<100:  l.mtr_smsz = "early_ketchup"
      else:         l.mtr_smsz = l.org_mtr_smsz
    elif l.mtr_schedule=="early_natto":
      if count<100:  l.mtr_smsz = "early_natto"
      else:         l.mtr_smsz = l.org_mtr_smsz
    else:
      raise(Exception("Invalid mtr_schedule"))
      

  def LogDPL(l, count):
    SaveYAML(l.dpl.MM.Save(l.dpl.MM.Options['base_dir']), l.dpl.MM.Options['base_dir']+'model_mngr.yaml')
    SaveYAML(l.dpl.DB.Save(), l.logdir+'database.yaml')
    SaveYAML(l.dpl.Save(), l.logdir+'dpl.yaml')

    config= {key: getattr(l.config,key) for key in l.config.__slots__}
    l.config_log = [config]
    # l.config_log.append(config)
    # SaveYAML(l.config_log, l.logdir+'config_log.yaml', interactive=False)

    # if l.restarting==True or count>1: w_mode = "a"
    # else: w_mode = "w"
    w_mode = "a"
    OpenW(l.logdir+'config_log.yaml',mode=w_mode,interactive=False).write(yamldump(ToStdType(l.config_log,lambda y:y), Dumper=YDumper))

    fp = open(l.logdir+'dpl_est.dat',w_mode)
    a_spill = l.dpl.DB.Entry[-1].Seq[-1].XS["a_spill2"].X[0][0]
    a_pour = l.dpl.DB.Entry[-1].Seq[-1].XS["a_pour"].X[0][0]
    env_return = (-100.0*max(0.0,0.3-a_pour)**2 - 1.0*max(0.0,a_pour-0.3)**2 - 1.0*max(0.0,a_spill)**2).item()
    values = [env_return] + sum([[true_return, l.dpl.Value(tree)] for true_return, tree in zip(l.true_return, l.node_best_tree)], [])
    idx = len(l.dpl.DB.Entry)-1
    fp.write('%i %s\n' % (idx, ' '.join(map(str,values))))
    fp.close()
    if w_mode=="a": CPrint(1,'Generated:',l.logdir+'dpl_est.dat')
    else: CPrint(1,'Added:',l.logdir+'dpl_est.dat')

    if not os.path.exists(l.logdir+"best_est_trees"): 
      os.mkdir(l.logdir+"best_est_trees")
    for i,tree in enumerate(l.node_best_tree):
      if i==0: joblib.dump(tree, l.logdir+"best_est_trees/"+"ep"+str(len(l.dpl.DB.Entry)-1)+"_n0.jb")
      else: joblib.dump(tree, l.logdir+"best_est_trees/"+"ep"+str(len(l.dpl.DB.Entry)-1)+"_n2a_"+str(i)+".jb")
    
    # #'''
    # #Analyze l.dpl.DB.Entry:
    # ptree= l.dpl.GetPTree('n0', {})
    # fp= open(l.logdir+'dpl_est.dat','w')
    # for i,eps in enumerate(l.dpl.DB.Entry):
    #   n0_0= eps.Find(('n0',0))[0]
    #   if n0_0 is None or eps.R is None:
    #     CPrint(4, 'l.dpl.DB has a broken entry')
    #     continue
    #   ptree.StartNode.XS= n0_0.XS
    #   ptree.ResetFlags()
    #   values= [eps.R, l.dpl.Value(ptree)]
    #   fp.write('%i %s\n' % (i, ' '.join(map(str,values))))
    # fp.close()
    # CPrint(1,'Generated:',l.logdir+'dpl_est.dat')
    # #'''


  # if l.interactive and 'log_dpl' in ct.__dict__ and (CPrint(1,'Restart from existing DPL?'), AskYesNo())[1]:
  if 'log_dpl' in ct.__dict__ and (CPrint(1,'Restart from existing DPL?'), AskYesNo())[1]:
    l.dpl= ct.log_dpl
    l.restarting= True
  else:
    mm_options= {
      'type': l.type,
      'base_dir': l.logdir+'models/',
      }
    mm= TModelManager(domain.SpaceDefs, domain.Models)
    mm.Load({'options':mm_options})
    if l.opt_conf['model_dir'] not in ('',None):
      if os.path.exists(l.opt_conf['model_dir']+'model_mngr.yaml'):
        mm.Load(LoadYAML(l.opt_conf['model_dir']+'model_mngr.yaml'), l.opt_conf['model_dir'])
      if l.opt_conf['model_dir_persistent']:
        mm.Options['base_dir']= l.opt_conf['model_dir']
      else:
        mm.Options['base_dir']= mm_options['base_dir']
    db= TGraphEpisodeDB()
    if l.opt_conf['db_src'] not in ('',None):
      db.Load(LoadYAML(l.opt_conf['db_src']))

    l.dpl= TGraphDynPlanLearn(domain, db, mm)
    l.restarting= False

  dpl_options={
    'base_dir': l.logdir,
    }
  InsertDict(dpl_options, l.opt_conf['dpl_options'])
  l.dpl.Load({'options':dpl_options})


  if not l.restarting:
    l.dpl.MM.Init()
    l.dpl.Init()

  ct.log_dpl= l.dpl  #for log purpose

  print 'Copying',PycToPy(__file__),'to',PycToPy(l.logdir+os.path.basename(__file__))
  CopyFile(PycToPy(__file__),PycToPy(l.logdir+os.path.basename(__file__)))

  count= 0
  # count = len(l.dpl.DB.Entry)
  if l.restarting:
    fp= OpenW(l.logdir+'dpl_log.dat','a', l.interactive)
  else:
    fp= OpenW(l.logdir+'dpl_log.dat','w', l.interactive)
    if len(l.dpl.DB.Entry)>0:
      for i in range(len(l.dpl.DB.Entry)):
        fp.write(l.dpl.DB.DumpOneYAML(i))
      fp.flush()
  
  l.priority_sampling = False
  while True:
    CPrint(2,'========== Start %4i =========='%count)
    EpisodicCallback(l,count)
    CPrint(3,"learning data size:",len(l.dpl.MM.Models["Fmvtopour2"][2].DataX))
    l.dpl.NewEpisode()
    l.user_viz= []
    l.node_best_tree = []
    l.true_return = []

    if l.priority_sampling==True:
      if t_sampling<l.max_priority_sampling:
        l.mtr_smsz = "latest_mtr_smsz"
        t_sampling += 1
      else:
        l.mtr_smsz = l.org_mtr_smsz
        l.priority_sampling = False

    try:
      Execute(ct,l)
    finally:
      ct.sim.StopPubSub(ct,l)
      ct.sim_local.sensor_callback= None
      ct.srvp.ode_pause()
    #l.sm_logfp.close()
    l.dpl.EndEpisode()
    CPrint(2,'========== End %4i =========='%count)
    #xyar_line= l.dpl.DB.Entry[-1].Dump()
    fp.write(l.dpl.DB.DumpOneYAML())
    fp.flush()
    CPrint(1,count,l.dpl.DB.DumpOne())
    count+= 1
    if l.dpl.DB.Entry[-1].R<l.return_epsiron: 
      l.priority_sampling = True
      t_sampling = 0
      # if l.priority_sampling==False:
      #   l.priority_sampling = True
      #   t_sampling = 0
      # else:
      #   pass

    LogDPL(l, count)
    if count>=l.num_episodes:  break
    if l.interactive:
      print 'Continue?'
      if not AskYesNo():  break

    # else:
    #   l.not_learn = True                #only sampling
    #   l.planning_node = []              #not plan
    #   l.mtr_smsz = "latest_mtr_smsz"    #use latest mtr and smsz

    #   def random_policy(l):
    #     l.reserve = dict()
    #     pc_rcv= np.array(l.xs.n0['ps_rcv'].X).reshape(4,3).mean(axis=0)
    #     l.reserve['gh_ratio']= SSA([Rand(0.0,1.0)])
    #     l.reserve['p_pour_trg0']= SSA(Vec([-0.3,0.35])+Vec([pc_rcv[0],pc_rcv[2]]))  #A bit above of p_pour_trg
    #     l.reserve['p_pour_trg']= SSA(Vec([Rand(0.2,1.2),Rand(0.1,0.7)]))
    #     l.reserve['dtheta1']= SSA([Rand(0.01,0.02)])
    #     if l.pour_skill=="std_pour":
    #       l.reserve['dtheta2']= SSA([Rand(0.002,0.005)])
    #     elif l.pour_skill=="shake_A":
    #       l.reserve['shake_spd']= SSA([Rand(0.7,0.9)])
    #       l.reserve['shake_axis2']= SSA([Rand(0.05,0.1),Rand(-0.5*math.pi,0.5*math.pi)])
    #     elif l.pour_skill=="choose":
    #       l.reserve['dtheta2']= SSA([Rand(0.002,0.005)])
    #       l.reserve['shake_spd']= SSA([Rand(0.7,0.9)])
    #       l.reserve['shake_axis2']= SSA([Rand(0.05,0.1),Rand(-0.5*math.pi,0.5*math.pi)])
    #       l.reserve['skill']= SSA([random.randint(0,1)])
    #     return l
      
    #   def bo_policy(l,A,r_list):
    #     l.reserve = dict()
    #     pc_rcv= np.array(l.xs.n0['ps_rcv'].X).reshape(4,3).mean(axis=0)
    #     if len(A)<1:
    #       l = random_policy(l)
    #     else:
    #       A = np.array(A)
    #       domain =[ {'name': 'gh_ratio', 'type': 'continuous', 'domain': (0,1)},
    #                 {'name': 'p_pour_trg0x', 'type': 'continuous', 'domain': (-0.3+pc_rcv[0],-0.3+pc_rcv[0])},
    #                 {'name': 'p_pour_trg0z', 'type': 'continuous', 'domain': (0.35+pc_rcv[2],0.35+pc_rcv[2])},
    #                 {'name': 'p_pour_trgx', 'type': 'continuous', 'domain': (0.2,1.2)},
    #                 {'name': 'p_pour_trgz', 'type': 'continuous', 'domain': (0.1,0.7)},
    #                 {'name': 'dtheta1', 'type': 'continuous', 'domain': (0.01,0.02)}]
    #       if l.pour_skill=="std_pour":
    #         domain += [{'name': 'dtheta2', 'type': 'continuous', 'domain': (0.002,0.005)}]
    #       elif l.pour_skill=="shake_A":
    #         domain += [ {'name': 'shake_spd', 'type': 'continuous', 'domain': (0.7,0.9)},
    #                     {'name': 'shake_axis2x', 'type': 'continuous', 'domain': (0.05,0.1)},
    #                     {'name': 'shake_axis2z', 'type': 'continuous', 'domain': (-0.5*math.pi,0.5*math.pi)}]
    #       # SHOULD BE FIXED
    #       elif l.pour_skill=="choose":
    #         domain += [ {'name': 'dtheta2', 'type': 'continuous', 'domain': (0.002,0.005)},
    #                     {'name': 'shake_spd', 'type': 'continuous', 'domain': (0.7,0.9)},
    #                     {'name': 'shake_axis2x', 'type': 'continuous', 'domain': (0.05,0.1)},
    #                     {'name': 'shake_axis2z', 'type': 'continuous', 'domain': (-0.5*math.pi,0.5*math.pi)},
    #                     {'name': 'skill', 'type': 'discrete', 'domain': (0,1)}] #SHOULD BE FIXED
    #       bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=domain, X=A, Y=r_list)
    #       a_list_next = bo_step.suggest_next_locations()
    #       print(a_list_next)
    #       hoge
    #     return l
      
    #   A = []
    #   r_list = []
    #   for i in range(l.max_priority_sampling):
    #     CPrint(3,'========== Start %s sampling %4i-%2i =========='%(l.sampling_mode,count,i))
    #     CPrint(3,"learning data size:",len(l.dpl.MM.Models["Fgrasp"][2].DataX))

    #     if l.sampling_mode=="random":   
    #       l = random_policy(l)
    #     elif l.sampling_mode=="bo":
    #       l = bo_policy(l,A,r_list)
    #     a_list = [
    #       l.reserve['gh_ratio'].X.tolist()[0][0],
    #       l.reserve['p_pour_trg0'].X.tolist()[0][0],
    #       l.reserve['p_pour_trg0'].X.tolist()[1][0],
    #       l.reserve['p_pour_trg'].X.tolist()[0][0],
    #       l.reserve['p_pour_trg'].X.tolist()[1][0],
    #       l.reserve['dtheta1'].X.tolist()[0][0]
    #     ]
    #     if l.pour_skill=="std_pour": a_list += [l.reserve['dtheta2'].X.tolist()[0][0]]
    #     elif l.pour_skill=="shake_A": a_list += [l.reserve['shake_spd'].X.tolist()[0][0]] \
    #                                           + [l.reserve['shake_axis2'].X.tolist()[0][0]] \
    #                                           + [l.reserve['shake_axis2'].X.tolist()[1][0]]
    #     #SHOULD BE FIXED
    #     elif l.pour_skill=="choose": a_list += [l.reserve['dtheta2'].X.tolist()[0][0]] \
    #                                           + [l.reserve['shake_spd'].X.tolist()[0][0]] \
    #                                           + [l.reserve['shake_axis2'].X.tolist()[0][0]] \
    #                                           + [l.reserve['shake_axis2'].X.tolist()[1][0]] \
    #                                           + [l.reserve['skill'].X.tolist()] #SHOULD BE FIXED
    #     A.append(a_list)

    #     l.dpl.NewEpisode()
    #     try:
    #       Execute(ct,l)
    #     finally:
    #       ct.sim.StopPubSub(ct,l)
    #       ct.sim_local.sensor_callback= None
    #       ct.srvp.ode_pause()
    #     l.dpl.EndEpisode()
    #     r_list.append([l.dpl.DB.Entry[-1].R])

    #     fp.write(l.dpl.DB.DumpOneYAML())
    #     fp.flush()
    #     config= {key: getattr(l.config,key) for key in l.config.__slots__}
    #     l.config_log = [config]
    #     OpenW(l.logdir+'config_log.yaml',mode="a",interactive=False).write(yamldump(ToStdType(l.config_log,lambda y:y), Dumper=YDumper))
        
    #     Print(a_list,r_list[-1])
    #     CPrint(3,'========== End %s sampling %4i-%2i =========='%(l.sampling_mode,count,i))

    #   l.not_learn = l.org_not_learn
    #   l.planning_node = l.org_planning_node
    #   l.mtr_smsz = l.org_mtr_smsz
    #   l.priority_sampling = False
  
  fp.close()

  ## for dpl_est debug!
  # tree = joblib.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/mtr_sms_sv/learn/shake_A/random/0055/normal/best_est_trees/ep203_n0.jb")
  # Print(l.dpl.Value(tree))
  # ct.Run('mysim.setup.setup_sv', l)
  # ptree= l.dpl.GetPTree('n0', {})
  # for i,eps in enumerate(l.dpl.DB.Entry):
  #   if i==199:
  #     n0_0= eps.Find(('n0',0))[0]
  #     if n0_0 is None or eps.R is None:
  #       CPrint(4, 'l.dpl.DB has a broken entry')
  #       continue
  #     ptree.StartNode.XS= n0_0.XS
  #     ptree.ResetFlags()
  #     # Print(ptree.Tree.keys())
  #     # Print(ptree.Tree[ptree.Start].XS)
  #     # ptree = l.dpl.ForwardTree(ptree, with_grad=True)
  #     for key in ptree.Tree.keys():
  #       Print(key,"J:",ptree.Tree[key].XS)
  #     l.dpl.BackwardTree(ptree)
  #     print("-"*20)
  #     for key in ptree.Tree.keys():
  #       Print(key,"J:",ptree.Tree[key].XS)
  #       # Print(key,"J:",ptree.Tree[key].XS)
  #     # Print(ptree.Start)
  #     # Print(ptree.Tree[ptree.Start].dFd)
  #     # values= [eps.R, l.dpl.Value(ptree)]
  # # Print("estimate147 before add ep:", values[1])

  # ct.Run('mysim.setup.setup_sv', l)
  # l.dpl.NewEpisode()
  # l.user_viz= []
  # Execute(ct,l)

  # ptree= l.dpl.GetPTree('n0', {})
  # for i,eps in enumerate(l.dpl.DB.Entry):
  #   if i==147:
  #     n0_0= eps.Find(('n0',0))[0]
  #     if n0_0 is None or eps.R is None:
  #       CPrint(4, 'l.dpl.DB has a broken entry')
  #       continue
  #     ptree.StartNode.XS= n0_0.XS
  #     ptree.ResetFlags()
  #     # Print(ptree.Tree[ptree.Start].XS)
  #     l.dpl.BackwardTree(ptree)
  #     for key in ptree.Tree.keys():
  #       Print(key,"J:",ptree.Tree[key].J)
  #       # Print(key,"J:",ptree.Tree[key].XS)
  #     values= [eps.R, l.dpl.Value(ptree)]
  # Print("estimate147 after add ep:", values[1])
  ###

  l= None
  return True
