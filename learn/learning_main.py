#!/usr/bin/python
from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
from matplotlib import pyplot as plt
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

def MergeData(l,dynamics):
  i = 0
  log_model_path = l.logdir+"/models/"
  if not os.path.exists(log_model_path):
    os.makedirs(log_model_path)
  f = open(log_model_path+dynamics+"_training_data.txt", mode='w')
  for data_dir in l.data_dir_list:
    dataX_path = data_dir+dynamics+"_"+"nn_data_x.dat"
    dataY_path = data_dir+dynamics+"_"+"nn_data_y.dat"
    try: 
      dataX = np.array(pickle.load(open(dataX_path, 'rb')), np.float32)
      dataY = np.array(pickle.load(open(dataY_path, 'rb')), np.float32)
    except:
      continue
    CPrint(3,"Read dir:",data_dir)
    CPrint(3,"Get sample size:",len(dataX))
    f.write("Read dir: "+data_dir+"\n")
    f.write("Get sample size: "+str(len(dataX))+"\n")
    if i==0:
      dataX_merged = dataX
      dataY_merged = dataY
    else:
      dataX_merged = np.concatenate([dataX_merged, dataX])
      dataY_merged = np.concatenate([dataY_merged, dataY])
    i += 1
    print("-------")
  CPrint(3,"Total sample size:",len(dataX_merged))
  f.write("Total sample size: "+str(len(dataX_merged))+"\n")
  f.close()
  return dataX_merged, dataY_merged

def Execute(l):
  # dynamics_list = ['Fgrasp']
  dynamics_list = l.dynamics_list
  for dynamics in dynamics_list:
    CPrint(2,"===== Train",dynamics,"=====")
    model = l.dpl.MM.Models[dynamics][2]
    # nn_options = l.nn_options
    model.Options.update(l.nn_options[dynamics])
    model.DataX, model.DataY = MergeData(l,dynamics)

    CPrint(3,"DataX shape:",model.DataX.shape)
    CPrint(3,"DataY shape:",model.DataY.shape)

    if l.do_update:
      model.Update(None,None,not_learn=l.not_learn)

def Run(ct,*args):
  l= TContainer(debug=True)
  l.logdir= args[0]
  l.opt_conf= args[1]
  l.nn_options = args[2]
  l.data_dir_list = args[3]
  l.dynamics_list = args[4]
  l.do_update = args[5]

  l.interactive= l.opt_conf['interactive']
  l.not_learn= l.opt_conf['not_learn']
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
    'dtheta1': SP('action',1,min=[0.01],max=[0.02]),  #Pouring skill parameter for all skills
    'dtheta2': SP('action',1,min=[0.002],max=[0.005]),  #Pouring skill parameter for 'std_pour'
    #'dtheta1': SP('state',1),  #Pouring skill parameter for all skills
    #'dtheta2': SP('state',1),  #Pouring skill parameter for 'std_pour'
    'shake_spd': SP('action',1,min=[0.7],max=[0.9]),  #Pouring skill parameter for 'shake_A'
    #'shake_spd': SP('state',1),  #Pouring skill parameter for 'shake_A'
    #'shake_axis': SP('action',2,min=[0.0,0.0],max=[0.1,0.1]),  #Pouring skill parameter for 'shake_A'
    'shake_axis2': SP('action',2,min=[0.05,-0.5*math.pi],max=[0.1,0.5*math.pi]),  #Pouring skill parameter for 'shake_A'
    #'shake_axis2': SP('state',2),  #Pouring skill parameter for 'shake_A'
    'shake_spd_B': SP('action',1,min=[2.0],max=[8.0]),  #Pouring skill parameter for 'shake_B'
    "shake_range" : SP('action',1,min=[0.02],max=[0.06]),  #Pouring skill parameter for 'shake_B'
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
    'Fflowc_shakeB10': [  #Flow control with shake_B.
      ['gh_abs','lp_pour',  #Removed 'p_pour_trg0','p_pour_trg'
       'da_trg','size_srcmouth','material2',
       'dtheta1','shake_spd_B','shake_range'],
      ['da_total','lp_flow','flow_var'],None],  #Removed 'p_pour'
    'Famount4': [  #Amount model common for tip and shake.
      ['lp_pour',  #Removed 'gh_abs','p_pour_trg0','p_pour_trg'
       'da_trg','material2',  #Removed 'size_srcmouth'
       'da_total','lp_flow','flow_var'],
      ['da_pour','da_spill2'],None],
    'Rrcvmv':  [['dps_rcv','v_rcv'],[REWARD_KEY],TLocalQuad(13,lambda y:-(np.dot(y[:12],y[:12]) + y[12]*y[12]))],
    'Rmvtopour':  [['p_pour_trg','p_pour'],[REWARD_KEY],TLocalQuad(5,lambda y:-0.1*((y[0]-y[2])**2+(y[1]-y[4])**2))],
    'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
                  TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2)],
    # 'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
    #               TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 10.0*max(0.0,y[2])**2)],
    'P1': [[],[PROB_KEY], TLocalLinear(0,1,lambda x:[1.0],lambda x:[0.0])],
    'P2':  [[],[PROB_KEY], TLocalLinear(0,2,lambda x:[1.0]*2,lambda x:[0.0]*2)],
    'Pskill': [['skill'],[PROB_KEY], TLocalLinear(0,2,lambda s:Delta1(2,s[0]),lambda s:[0.0]*2)],
    }

  def LogDPL(l):
    SaveYAML(l.dpl.MM.Save(l.dpl.MM.Options['base_dir']), l.dpl.MM.Options['base_dir']+'model_mngr.yaml')
    SaveYAML(l.dpl.DB.Save(), l.logdir+'database.yaml')
    SaveYAML(l.dpl.Save(), l.logdir+'dpl.yaml')

  # if l.interactive and 'log_dpl' in ct.__dict__ and (CPrint(1,'Restart from existing DPL?'), AskYesNo())[1]:
  if 'log_dpl' in ct.__dict__ and (CPrint(1,'Restart from existing DPL?'), AskYesNo())[1]:
    l.dpl= ct.log_dpl
    l.restarting= True
  else:
    mm_options= {
      #'type': 'lwr',
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

  Execute(l)
  LogDPL(l)

  l= None
  return True
