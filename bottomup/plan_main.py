#!/usr/bin/python
from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
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

def RwdModel():
  modeldir= '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'\
            +'reward_model'+"/"
  FRwd= TNNRegression()
  prefix= modeldir+'p1_model/FRwd3'
  FRwd.Load(LoadYAML(prefix+'.yaml'), prefix)
  FRwd.Init()

  return FRwd

def PrintEstTree(l):
  tree = l.node_best_tree[-1]
  for key in tree.Tree.keys():
    print(key.A)
    print(tree.Tree[key].XS)

def Execute(ct,l, count):
  l.custom_smsz = l.custom_smsz_all[count]

  ct.Run('mysim.setup.setup_sv', l)
  sim= ct.sim
  #l= ct.sim_local

  #NOTE: Do not include 'da_trg' in obs_keys0 since 'da_trg' should be kept during some node transitions.
  obs_keys0= ('ps_rcv','p_pour','lp_pour','a_trg','size_srcmouth','material2')
  obs_keys_after_grab= obs_keys0+('gh_abs',)
  obs_keys_before_flow= obs_keys_after_grab+('a_pour','a_spill2','a_total')
  obs_keys_after_flow= obs_keys_before_flow+('lp_flow','flow_var','da_pour','da_spill2','da_total')

  l.xs= TContainer()  #l.xs.NODE= XSSA
  l.idb= TContainer()  #l.idb.NODE= index in DB

  with sim.TPause(ct):
    CPrint(2,'Node:','n0')
    l.xs.n0= ObserveXSSA(l,None,obs_keys0+('da_trg',))

    pc_rcv= np.array(l.xs.n0['ps_rcv'].X).reshape(4,3).mean(axis=0)  #Center of ps_rcv
    l.xs.n0['gh_ratio']= SSA([0.5])
    l.xs.n0['p_pour_trg0']= SSA(Vec([-0.3,0.35])+Vec([pc_rcv[0],pc_rcv[2]]))  #A bit above of p_pour_trg
    l.xs.n0['dtheta1']= SSA([0.014])
    l.xs.n0['dtheta2']= SSA([0.002])
    l.xs.n0['shake_spd']= SSA([0.8])
    # l.xs.n0['shake_axis2']= SSA([0.08,0.0])

    if l.pour_skill=="std_pour":
      l.xs.n0['skill']= SSA([0])
    elif l.pour_skill=="shake_A":
      l.xs.n0['skill']= SSA([1])
    
    # planed result into l.xs.n0
    res = l.dpl.Plan('n0', l.xs.n0, l.interactive)
    l.node_best_tree.append(res.PTree)


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
    # 'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position (x,z)
    'p_pour_trg': SP('action',2,min=[-0.1+0.6,0.325+0.202],max=[0.0+0.6,0.325+0.202]),  #Target pouring axis position (x,z)
    'dtheta1': SP('state',1,min=[0.01],max=[0.02]),  #Pouring skill parameter for all skills
    'dtheta2': SP('state',1,min=[0.002],max=[0.005]),  #Pouring skill parameter for 'std_pour'
    #'dtheta1': SP('state',1),  #Pouring skill parameter for all skills
    #'dtheta2': SP('state',1),  #Pouring skill parameter for 'std_pour'
    'shake_spd': SP('state',1,min=[0.7],max=[0.9]),  #Pouring skill parameter for 'shake_A'
    #'shake_spd': SP('state',1),  #Pouring skill parameter for 'shake_A'
    #'shake_axis': SP('action',2,min=[0.0,0.0],max=[0.1,0.1]),  #Pouring skill parameter for 'shake_A'
    'shake_axis2': SP('action',2,min=[0.028,-1.22],max=[0.028,-1.22]),  #Pouring skill parameter for 'shake_A'
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
      ['lp_pour','size_srcmouth'],
      ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    # 'Fflowc_shakeA10': [  #Flow control with shake_A.
    #   ['gh_abs','lp_pour',  #Removed 'p_pour_trg0','p_pour_trg'
    #    'da_trg','size_srcmouth','material2',
    #    'dtheta1','shake_spd','shake_axis2'],
      # ['da_total','lp_flow','flow_var'],None],  #Removed 'p_pour'
    'Fflowc_shakeA10': [  #Flow control with shake_A.
      ['lp_pour','size_srcmouth','shake_axis2'],
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
    # 'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
    #               TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2)],
    'Rdamount':  [['da_pour'],[REWARD_KEY],
                  RwdModel()],
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
  #   domain.SpaceDefs.update({'skill': SP('state',num=2)})
    # domain.Graph.update({'n2c': TDynNode('n2b','P1',('Fnone','n3ti'))})
  # elif l.pour_skill=="shake_A":
  #   domain.SpaceDefs.update({'skill': SP('state',num=2)})
    # domain.Graph.update({'n2c': TDynNode('n2b','P1',('Fnone','n3sa'))})
  # elif l.pour_skill=="choose":
  #   domain.Graph.update({'n2c': TDynNode('n2b','Pskill',('Fnone','n3ti'),('Fnone','n3sa'))})
  
  #Learning scheduling
  def EpisodicCallback(l,count):
    Rdamount_default= [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
          TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2)]
    Rdamount_amount= [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
          TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2)]
    Rdamount_early_tip= [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
          TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2 - (200.0 if y[3]!=0 else 0.0))]
    Rdamount_early_shakeA= [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
          TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2 - (200.0 if y[3]!=1 else 0.0))]
    Rdamount_tip_amount= [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
          TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - (200.0 if y[3]!=0 else 0.0))]
    Rdamount_shakeA_amount= [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
          TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - (200.0 if y[3]!=1 else 0.0))]
    #'rwd_schedule': None,  #None, 'early_tip', 'early_shakeA'
    if l.rwd_schedule is None:
      #No reward scheduling
      pass
    elif l.rwd_schedule=='early_tip':
      #Reward scheduling (FOR EARLY TIPPING)
      if count<300:  l.dpl.d.Models['Rdamount']= Rdamount_early_tip
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
    elif l.rwd_schedule=="only_tip_only_amount": l.dpl.d.Models['Rdamount']= Rdamount_tip_amount
    elif l.rwd_schedule=="only_shake_only_amount": l.dpl.d.Models['Rdamount']= Rdamount_shakeA_amount
    elif l.rwd_schedule=="only_amount": l.dpl.d.Models['Rdamount']= Rdamount_amount

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
    
    # l.dpl.d.Models['Rdamount'][2].Load(data={"options": {"tune_h": True, "maxd1": 1e10, "maxd2": 1e10}})
    # l.dpl.d.Models['Rdamount'] = [['da_pour'],[REWARD_KEY],RwdModel()]



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
  
  count= 0
  l.priority_sampling = False
  while True:
    CPrint(2,'========== Start %4i =========='%count)
    EpisodicCallback(l,count)
    CPrint(3,"learning data size:",len(l.dpl.MM.Models["Fmvtopour2"][2].DataX))
    l.dpl.NewEpisode()
    l.user_viz= []
    l.node_best_tree = []

    if l.priority_sampling==True:
      if t_sampling<l.max_priority_sampling:
        l.mtr_smsz = "latest_mtr_smsz"
        t_sampling += 1
      else:
        l.mtr_smsz = l.org_mtr_smsz
        l.priority_sampling = False

    try:
      Execute(ct,l,count)
    finally:
      ct.sim.StopPubSub(ct,l)
      ct.sim_local.sensor_callback= None
      ct.srvp.ode_pause()
    #l.sm_logfp.close()
    l.dpl.EndEpisode()

    PrintEstTree(l)

    if not os.path.exists(l.logdir+"best_est_trees"): 
      os.mkdir(l.logdir+"best_est_trees")
    for i,tree in enumerate(l.node_best_tree):
      if i==0: joblib.dump(tree, l.logdir+"best_est_trees/"+"ep"+str(len(l.dpl.DB.Entry)-1)+"_n0.jb")
      else: joblib.dump(tree, l.logdir+"best_est_trees/"+"ep"+str(len(l.dpl.DB.Entry)-1)+"_n2a_"+str(i)+".jb")

    CPrint(2,'========== End %4i =========='%count)

    count += 1
    if count>=l.num_episodes:  break

  l= None
  return True