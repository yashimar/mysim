#!/usr/bin/python
from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
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
  ct.Run('tsim2.setup', l)
  sim= ct.sim
  #l= ct.sim_local

  actions={
    'grab'         : lambda a: ct.Run('tsim2.act.grab', a),
    'move_to_rcv'  : lambda a: ct.Run('tsim2.act.move_to_rcv', a),
    'move_to_pour' : lambda a: ct.Run('tsim2.act.move_to_pour', a),
    'std_pour'     : lambda a: ct.Run('mysim.act.std_pour', a),
    'shake_A'      : lambda a: ct.Run('mysim.act.shake_A_5s', a),
    }

  #NOTE: Do not include 'da_trg' in obs_keys0 since 'da_trg' should be kept during some node transitions.
  obs_keys0= ('ps_rcv','p_pour','lp_pour','a_trg','size_srcmouth','material2')
  obs_keys_after_grab= obs_keys0+('gh_abs',)
  obs_keys_before_flow= obs_keys_after_grab+('a_pour','a_spill2','a_total')
  obs_keys_after_flow= obs_keys_before_flow+('lp_flow','flow_var','da_pour','da_spill2','da_total')

  l.xs= TContainer()  #l.xs.NODE= XSSA
  l.idb= TContainer()  #l.idb.NODE= index in DB

  with sim.TPause(ct):  #Pause during plan/learn
    CPrint(2,'Node:','n0')
    l.xs.n0= ObserveXSSA(l,None,obs_keys0+('da_trg',))
    #TEST: Heuristic init guess
    pc_rcv= np.array(l.xs.n0['ps_rcv'].X).reshape(4,3).mean(axis=0)  #Center of ps_rcv
    #l.xs.n0['gh_ratio']= SSA([0.5])
    #l.xs.n0['p_pour_trg']= SSA(Vec([0.0,0.1])+Vec([pc_rcv[0],pc_rcv[2]]))  #A bit above of Center of ps_rcv
    l.xs.n0['p_pour_trg0']= SSA(Vec([-0.3,0.35])+Vec([pc_rcv[0],pc_rcv[2]]))  #A bit above of p_pour_trg
    #l.xs.n0['dtheta1']= SSA([0.014])
    #l.xs.n0['dtheta2']= SSA([0.004])
    #l.xs.n0['shake_spd']= SSA([0.8])
    #l.xs.n0['shake_axis2']= SSA([0.08,0.0])
    #l.xs.n0['skill']= SSA([1])
    res= l.dpl.Plan('n0', l.xs.n0, l.interactive)
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
    l.dpl.MM.Update('Fgrasp',l.xs.prev,l.xs.n1, not_learn=l.not_learn)
    #res= l.dpl.Plan('n1', l.xs.n1)
    l.idb.n1= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n1',xs=l.xs.n1)
    l.xs.prev= l.xs.n1
    l.idb.prev= l.idb.n1

  p_pour_trg0= ToList(l.xs.n1['p_pour_trg0'].X)
  p_pour_trg= ToList(l.xs.n1['p_pour_trg'].X)
  actions['move_to_rcv']({'p_pour_trg0':p_pour_trg0})
  VizPP(l,[p_pour_trg0[0],0.0,p_pour_trg0[1]],[0.,1.,0.])
  VizPP(l,[p_pour_trg[0],0.0,p_pour_trg[1]],[0.5,0.,1.])

  with sim.TPause(ct):  #Pause during plan/learn
    #Branch-1: reward
    CPrint(2,'Node:','n1rcvmv')
    l.xs.n1rcvmv= CopyXSSA(l.xs.prev)
    InsertDict(l.xs.n1rcvmv, ObserveXSSA(l,l.xs.prev,('dps_rcv','v_rcv')))
    l.dpl.MM.Update('Fmvtorcv_rcvmv',l.xs.prev,l.xs.n1rcvmv, not_learn=l.not_learn)
    #res= l.dpl.Plan('n1rcvmv', l.xs.n1rcvmv)
    l.idb.n1rcvmv= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n1rcvmv',xs=l.xs.n1rcvmv)

    CPrint(2,'Node:','n1rcvmvr')
    #Since we have 'Rrcvmv', we just use it to get the next XSSA
    l.xs.n1rcvmvr= l.dpl.Forward('Rrcvmv',l.xs.n1rcvmv)
    l.idb.n1rcvmvr= l.dpl.DB.AddToSeq(parent=l.idb.n1rcvmv,name='n1rcvmvr',xs=l.xs.n1rcvmvr)

    #Branch-2: main procedure
    CPrint(2,'Node:','n2a','(update)')
    l.xs.n2a= CopyXSSA(l.xs.prev)
    InsertDict(l.xs.n2a, ObserveXSSA(l,l.xs.prev,obs_keys_after_grab))
    l.dpl.MM.Update('Fmvtorcv',l.xs.prev,l.xs.n2a, not_learn=l.not_learn)

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
      InsertDict(l.xs.n2a, ObserveXSSA(l,l.xs.prev,obs_keys_after_grab+('da_trg',)))
      #TEST: Heuristic init guess
      #l.xs.n2a['skill']= SSA([1])
      # res= l.dpl.Plan('n2a', l.xs.n2a, l.interactive)
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
      l.dpl.MM.Update('Fmvtopour2',l.xs.prev,l.xs.n2b, not_learn=l.not_learn)
      #res= l.dpl.Plan('n2b', l.xs.n2b)
      l.idb.n2b= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n2b',xs=l.xs.n2b)
      l.xs.prev= l.xs.n2b
      l.idb.prev= l.idb.n2b

    #Branch-1: main procedure
    #Just go to 'n2c'

    #Branch-2: reward
    CPrint(2,'Node:','n2br')
    #Since we have 'Rmvtopour', we just use it to get the next XSSA
    l.xs.n2br= l.dpl.Forward('Rmvtopour',l.xs.prev)
    l.idb.n2br= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n2br',xs=l.xs.n2br)

    with sim.TPause(ct):  #Pause during plan/learn
      #Plan l.selected_skill from ('std_pour','shake_A','shake_B')
      CPrint(2,'Node:','n2c')
      l.xs.n2c= CopyXSSA(l.xs.prev)
      InsertDict(l.xs.n2c, ObserveXSSA(l,l.xs.prev,obs_keys_before_flow))
      #l.dpl.MM.Update('Fnone',l.xs.prev,l.xs.n2c, not_learn=l.not_learn)
      #res= l.dpl.Plan('n2c', l.xs.n2c)
      l.idb.n2c= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n2c',xs=l.xs.n2c)
      l.xs.prev= l.xs.n2c
      l.idb.prev= l.idb.n2c

    if l.skill_pour=="std_pour": selected_skill=l.skill_pour
    elif l.skill_pour=="shake_A": selected_skill=l.skill_pour
    elif l.skill_pour=="choose":
      idx = int(l.xs.n2c['skill'].X[0])
      selected_skill= ('std_pour','shake_A')[idx]
    # selected_skill= ('std_pour','shake_A')[l.xs.n2c['skill'].X[0]]
    #selected_skill= 'shake_A'
    if selected_skill=='std_pour':
      dtheta1= l.xs.n2c['dtheta1'].X[0,0]
      dtheta2= l.xs.n2c['dtheta2'].X[0,0]
      actions['std_pour']({'dtheta1':dtheta1, 'dtheta2':dtheta2})

      with sim.TPause(ct):  #Pause during plan/learn
        CPrint(2,'Node:','n3ti')
        l.xs.n3ti= CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n3ti, ObserveXSSA(l,l.xs.prev,obs_keys_after_flow))
        l.dpl.MM.Update('Fflowc_tip10',l.xs.prev,l.xs.n3ti, not_learn=l.not_learn)
        #res= l.dpl.Plan('n3ti', l.xs.n3ti)
        l.idb.n3ti= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n3ti',xs=l.xs.n3ti)
        l.xs.prev= l.xs.n3ti
        l.idb.prev= l.idb.n3ti

        CPrint(2,'Node:','n4ti')
        l.xs.n4ti= CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n4ti, ObserveXSSA(l,l.xs.prev,()))  #Observation is omitted since there is no change
        #WARNING:NOTE: Famount4 uses 'lp_pour' as input, so here we use a trick:
        xs_in= CopyXSSA(l.xs.prev)
        xs_in['lp_pour']= l.xs.n2c['lp_pour']
        #l.dpl.MM.Update('Famount4',l.xs.prev,l.xs.n4ti, not_learn=l.not_learn)
        l.dpl.MM.Update('Famount4',xs_in,l.xs.n4ti, not_learn=l.not_learn)
        #res= l.dpl.Plan('n4ti', l.xs.n4ti)
        l.idb.n4ti= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n4ti',xs=l.xs.n4ti)
        l.xs.prev= l.xs.n4ti
        l.idb.prev= l.idb.n4ti

        CPrint(2,'Node:','n4tir')
        l.xs.n4tir= l.dpl.Forward('Rdamount',l.xs.prev)
        l.idb.n4tir= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n4tir',xs=l.xs.n4tir)

    elif selected_skill=='shake_A':
      dtheta1= l.xs.n2c['dtheta1'].X[0,0]
      shake_spd= l.xs.n2c['shake_spd'].X[0,0]
      shake_axis2= ToList(l.xs.n2c['shake_axis2'].X)
      actions['shake_A']({'dtheta1':dtheta1, 'shake_spd':shake_spd, 'shake_axis2':shake_axis2})

      with sim.TPause(ct):  #Pause during plan/learn
        CPrint(2,'Node:','n3sa')
        l.xs.n3sa= CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n3sa, ObserveXSSA(l,l.xs.prev,obs_keys_after_flow))
        l.dpl.MM.Update('Fflowc_shakeA10',l.xs.prev,l.xs.n3sa, not_learn=l.not_learn)
        #res= l.dpl.Plan('n3sa', l.xs.n3sa)
        l.idb.n3sa= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n3sa',xs=l.xs.n3sa)
        l.xs.prev= l.xs.n3sa
        l.idb.prev= l.idb.n3sa

        CPrint(2,'Node:','n4sa')
        l.xs.n4sa= CopyXSSA(l.xs.prev)
        InsertDict(l.xs.n4sa, ObserveXSSA(l,l.xs.prev,()))  #Observation is omitted since there is no change
        #WARNING:NOTE: Famount4 uses 'lp_pour' as input, so here we use a trick:
        xs_in= CopyXSSA(l.xs.prev)
        xs_in['lp_pour']= l.xs.n2c['lp_pour']
        #l.dpl.MM.Update('Famount4',l.xs.prev,l.xs.n4sa, not_learn=l.not_learn)
        l.dpl.MM.Update('Famount4',xs_in,l.xs.n4sa, not_learn=l.not_learn)
        #res= l.dpl.Plan('n4sa', l.xs.n4sa)
        l.idb.n4sa= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n4sa',xs=l.xs.n4sa)
        l.xs.prev= l.xs.n4sa
        l.idb.prev= l.idb.n4sa

        CPrint(2,'Node:','n4sar')
        l.xs.n4sar= l.dpl.Forward('Rdamount',l.xs.prev)
        l.idb.n4sar= l.dpl.DB.AddToSeq(parent=l.idb.prev,name='n4sar',xs=l.xs.n4sar)

    # # Conditions to break the try-and-error loop
    # if l.IsPoured():
    #   break
    # if l.IsTimeout() or l.IsEmpty():  # or l.IsSpilled()
    #   break
    # if not IsSuccess(l.exec_status):
    #   break

    # repeated= True
    break

def Run(ct,*args):
  l = args[0]
  l.config_callback = args[1]
  l.logdir= args[2]
  l.opt_conf = args[3]
  l.ask_restart = args[4]
  l.reward_func = args[5]
  l.pour_skills = args[6]
  l.count = args[7]

  l.skill_pour = l.opt_conf["skill_pour"]
  l.interactive= l.opt_conf['interactive']
  l.num_episodes= l.opt_conf['num_episodes']
  l.num_log_interval= l.opt_conf['num_log_interval']
  l.rwd_schedule= l.opt_conf['rwd_schedule']

  l.not_learn= l.opt_conf['not_learn']
  l.config_log = []
  #l.not_learn= True  #Models are not trained.

  #Setup dynamic planner/learner
  domain= TGraphDynDomain()
  SP= TCompSpaceDef
  domain.SpaceDefs={
    'skill': SP('select',num=2),  #Skill selection
    'ps_rcv': SP('state',12),  #4 edge point positions (x,y,z)*4 of receiver
    'gh_ratio': SP('action',1,min=[0.0],max=[1.0]),  #Gripper height (ratio)
    'gh_abs': SP('state',1),  #Gripper height (absolute value)
    'p_pour_trg0': SP('state',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position of preparation before pouring (x,z)
      #NOTE: we stopped to plan p_pour_trg0
    'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position (x,z)
    'dtheta1': l.pour_skills["dtheta1"],  #Pouring skill parameter for all skills
    'dtheta2': l.pour_skills["dtheta2"],  #Pouring skill parameter for 'std_pour'
    #'dtheta1': SP('state',1),  #Pouring skill parameter for all skills
    #'dtheta2': SP('state',1),  #Pouring skill parameter for 'std_pour'
    'shake_spd': l.pour_skills["shake_spd"],  #Pouring skill parameter for 'shake_A'
    #'shake_spd': SP('state',1),  #Pouring skill parameter for 'shake_A'
    #'shake_axis': SP('action',2,min=[0.0,0.0],max=[0.1,0.1]),  #Pouring skill parameter for 'shake_A'
    'shake_axis2': l.pour_skills["shake_axis2"],  #Pouring skill parameter for 'shake_A'
    #'shake_axis2': SP('state',2),  #Pouring skill parameter for 'shake_A'
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
    'Fgrasp': [['gh_ratio'],['gh_abs'],None],  #Grasping. NOTE: removed ps_rcv
    'Fmvtorcv': [  #Move to receiver
      ['ps_rcv','gh_abs','p_pour','p_pour_trg0'],
      ['ps_rcv','p_pour'],None],
    'Fmvtorcv_rcvmv': [  #Move to receiver: receiver movement
      ['ps_rcv','gh_abs','p_pour','p_pour_trg0'],
      ['dps_rcv','v_rcv'],None],
    'Fmvtopour2': [  #Move to pouring point
      ['ps_rcv','gh_abs','p_pour','p_pour_trg'],
      ['lp_pour'],None],
    'Fflowc_tip10': [  #Flow control with tipping.
      ['gh_abs','lp_pour',  #Removed 'p_pour_trg0','p_pour_trg'
       'da_trg','size_srcmouth','material2',
       'dtheta1','dtheta2'],
      ['da_total','lp_flow','flow_var'],None],  #Removed 'p_pour'
    'Fflowc_shakeA10': [  #Flow control with shake_A.
      ['gh_abs','lp_pour',  #Removed 'p_pour_trg0','p_pour_trg'
       'da_trg','size_srcmouth','material2',
       'dtheta1','shake_spd','shake_axis2'],
      ['da_total','lp_flow','flow_var'],None],  #Removed 'p_pour'
    'Famount4': [  #Amount model common for tip and shake.
      ['lp_pour',  #Removed 'gh_abs','p_pour_trg0','p_pour_trg'
       'da_trg','material2',  #Removed 'size_srcmouth'
       'da_total','lp_flow','flow_var'],
      ['da_pour','da_spill2'],None],
    'Rrcvmv':  [['dps_rcv','v_rcv'],[REWARD_KEY],TLocalQuad(13,lambda y:-(np.dot(y[:12],y[:12]) + y[12]*y[12]))],
    'Rmvtopour':  [['p_pour_trg','p_pour'],[REWARD_KEY],TLocalQuad(5,lambda y:-0.1*((y[0]-y[2])**2+(y[1]-y[4])**2))],
    #'Ramount':  [['a_pour','a_trg','a_spill2'],[REWARD_KEY],TLocalQuad(3,lambda y:-100.0*(y[1]-y[0])*(y[1]-y[0]) - y[2]*y[2])],
    'Rdamount':  l.reward_func, 
    'P1': [[],[PROB_KEY], TLocalLinear(0,1,lambda x:[1.0],lambda x:[0.0])],
    'P2':  [[],[PROB_KEY], TLocalLinear(0,2,lambda x:[1.0]*2,lambda x:[0.0]*2)],
    'Pskill': [['skill'],[PROB_KEY], TLocalLinear(0,2,lambda s:Delta1(2,s[0]),lambda s:[0.0]*2)],
    }
  if l.skill_pour=="std_pour":
    domain.Graph={
      'n0': TDynNode(None,'P1',('Fgrasp','n1')),
      'n1': TDynNode('n0','P2',('Fmvtorcv','n2a'),('Fmvtorcv_rcvmv','n1rcvmv')),
      'n1rcvmv': TDynNode('n1','P1',('Rrcvmv','n1rcvmvr')),
      'n1rcvmvr': TDynNode('n1rcvmv'),
      'n2a': TDynNode('n1','P1',('Fmvtopour2','n2b')),
      'n2b': TDynNode('n2a','P2',('Fnone','n2c'),('Rmvtopour','n2br')),
      'n2br': TDynNode('n2b'),
      'n2c': TDynNode('n2b','P1',('Fflowc_tip10','n3ti')),
      #Tipping:
      'n3ti': TDynNode('n2c','P1',('Famount4','n4ti')),
      'n4ti': TDynNode('n3ti','P1',('Rdamount','n4tir')),
      'n4tir': TDynNode('n4ti'),
    }
  elif l.skill_pour=="shake_A":
    domain.Graph={
      'n0': TDynNode(None,'P1',('Fgrasp','n1')),
      'n1': TDynNode('n0','P2',('Fmvtorcv','n2a'),('Fmvtorcv_rcvmv','n1rcvmv')),
      'n1rcvmv': TDynNode('n1','P1',('Rrcvmv','n1rcvmvr')),
      'n1rcvmvr': TDynNode('n1rcvmv'),
      'n2a': TDynNode('n1','P1',('Fmvtopour2','n2b')),
      'n2b': TDynNode('n2a','P2',('Fnone','n2c'),('Rmvtopour','n2br')),
      'n2br': TDynNode('n2b'),
      'n2c': TDynNode('n2b','P1',('Fflowc_shakeA10','n3sa')),
      #Shaking-A:
      'n3sa': TDynNode('n2c','P1',('Famount4','n4sa')),
      'n4sa': TDynNode('n3sa','P1',('Rdamount','n4sar')),
      'n4sar': TDynNode('n4sa'),
    }
  elif l.skill_pour=="choose":
    domain.Graph={
      'n0': TDynNode(None,'P1',('Fgrasp','n1')),
      'n1': TDynNode('n0','P2',('Fmvtorcv','n2a'),('Fmvtorcv_rcvmv','n1rcvmv')),
      'n1rcvmv': TDynNode('n1','P1',('Rrcvmv','n1rcvmvr')),
      'n1rcvmvr': TDynNode('n1rcvmv'),
      'n2a': TDynNode('n1','P1',('Fmvtopour2','n2b')),
      'n2b': TDynNode('n2a','P2',('Fnone','n2c'),('Rmvtopour','n2br')),
      'n2br': TDynNode('n2b'),
      'n2c': TDynNode('n2b','Pskill',('Fflowc_tip10','n3ti'),('Fflowc_shakeA10','n3sa')),
      #Tipping:
      'n3ti': TDynNode('n2c','P1',('Famount4','n4ti')),
      'n4ti': TDynNode('n3ti','P1',('Rdamount','n4tir')),
      'n4tir': TDynNode('n4ti'),
      #Shaking-A:
      'n3sa': TDynNode('n2c','P1',('Famount4','n4sa')),
      'n4sa': TDynNode('n3sa','P1',('Rdamount','n4sar')),
      'n4sar': TDynNode('n4sa'),
    }
  #Learning scheduling
  def EpisodicCallback(l,count):
    Rdamount_default= [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
          TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2)]
    Rdamount_early_tip= [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
          TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2 - (10.0 if y[3]!=0 else 0.0))]
    Rdamount_early_shakeA= [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
          TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2 - (10.0 if y[3]!=1 else 0.0))]
    #'rwd_schedule': None,  #None, 'early_tip', 'early_shakeA'
    if l.rwd_schedule is None:
      #No reward scheduling
      pass
    elif l.rwd_schedule=='early_tip':
      #Reward scheduling (FOR EARLY TIPPING)
      if count<10:  l.dpl.d.Models['Rdamount']= Rdamount_early_tip
      else:         l.dpl.d.Models['Rdamount']= Rdamount_default
    elif l.rwd_schedule=='early_shakeA':
      #Reward scheduling (FOR EARLY SHAKING-A)
      if count<10:  l.dpl.d.Models['Rdamount']= Rdamount_early_shakeA
      else:         l.dpl.d.Models['Rdamount']= Rdamount_default

  def LogDPL(l):
    SaveYAML(l.dpl.MM.Save(l.dpl.MM.Options['base_dir']), l.dpl.MM.Options['base_dir']+'model_mngr.yaml')
    SaveYAML(l.dpl.DB.Save(), l.logdir+'database.yaml')
    SaveYAML(l.dpl.Save(), l.logdir+'dpl.yaml')

    config= {key: getattr(l.config,key) for key in l.config.__slots__}
    l.config_log.append(config)
    SaveYAML(l.config_log, l.logdir+'config_log.yaml', interactive=False)

    #'''
    #Analyze l.dpl.DB.Entry:
    ptree= l.dpl.GetPTree('n0', {})
    if l.count==0:
      fp= open(l.logdir+'dpl_est.dat','w')
    else:
      fp= open(l.logdir+'dpl_est.dat','a')
    for i,eps in enumerate(l.dpl.DB.Entry):
      n0_0= eps.Find(('n0',0))[0]
      if n0_0 is None or eps.R is None:
        CPrint(4, 'l.dpl.DB has a broken entry')
        continue
      ptree.StartNode.XS= n0_0.XS
      ptree.ResetFlags()
      values= [eps.R, l.dpl.Value(ptree)]
      fp.write('%i %s\n' % (i, ' '.join(map(str,values))))
    fp.close()
    CPrint(1,'Generated:',l.logdir+'dpl_est.dat')
    #'''

  # if l.interactive and 'log_dpl' in ct.__dict__ and (CPrint(1,'Restart from existing DPL?'), AskYesNo())[1]:
  if l.ask_restart and 'log_dpl' in ct.__dict__ and (CPrint(1,'Restart from existing DPL?'), AskYesNo())[1]:
    l.dpl= ct.log_dpl
    l.restarting= True
  else:
    mm_options= {
      #'type': 'lwr',
      'base_dir': l.logdir+'models/',
      # "base_dir": '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/learn_dynamics2/models/'
      }
    mm= TModelManager(domain.SpaceDefs, domain.Models)
    mm.Load({'options':mm_options})
    if l.opt_conf['model_dir'] not in ('',None):
      if os.path.exists(l.opt_conf['model_dir']["main"]+'model_mngr.yaml'):
        CPrint(3,"Load Options:",l.opt_conf['model_dir']["main"]+'model_mngr.yaml')
        mm.Load(LoadYAML(l.opt_conf['model_dir']["main"]+'model_mngr.yaml'))
        mm.CreateModels()
        for key in mm.Learning:
          In,Out,F= mm.Models[key]
          prefix,path= mm.GetFilePrefixPath(l.opt_conf['model_dir'][key],key)
          if os.path.exists(path):
            F.Load(LoadYAML(path), prefix)
            CPrint(3,"Find dynamics model:",key)
            Print(path)
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
  # if l.restarting:
  #   fp= OpenW(l.logdir+'dpl_log.dat','a', l.interactive)
  # else:
  #   fp= OpenW(l.logdir+'dpl_log.dat','w', l.interactive)
  #   if len(l.dpl.DB.Entry)>0:
  #     for i in range(len(l.dpl.DB.Entry)):
  #       fp.write(l.dpl.DB.DumpOneYAML(i))
  #     fp.flush()
  if l.count!=0 and os.path.exists(l.logdir+'dpl_log.dat'):
        fp= OpenW(l.logdir+'dpl_log.dat','a', l.interactive)
  else:
    fp= OpenW(l.logdir+'dpl_log.dat','w', l.interactive)
    if len(l.dpl.DB.Entry)>0:
      for i in range(len(l.dpl.DB.Entry)):
        fp.write(l.dpl.DB.DumpOneYAML(i))
      fp.flush()
  while True:
    for i in range(l.num_log_interval):
      CPrint(2,'========== Start %4i =========='%count)
      ###
      # l.dpl.MM.Init()
      # l.dpl.Init()
      # l.dpl.database = TGraphEpisodeDB()
      ###
      EpisodicCallback(l,count)
      l.dpl.NewEpisode()
      l.user_viz= []
      #l.sm_logfp= OpenW(l.logdir+'sm/sm_log%04i.dat'%count,'w')
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
      if count>=l.num_episodes:  break
    #ct.Run('tsim2.dplD20log',l)
    LogDPL(l)
    if count>=l.num_episodes:  break
    if l.interactive:
      print 'Continue?'
      if not AskYesNo():  break
  fp.close()

  l= None
  return True
