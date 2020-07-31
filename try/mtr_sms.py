from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *

def Help():
  pass

def ConfigCallback(ct,l,sim):
  m_setup= ct.Load('tsim2.setup')
  l.amount_trg= l._amount_trg
  l.spilled_stop= l._spilled_stop
  l.config.RcvPos= [0.6, l.config.RcvPos[1], l.config.RcvPos[2]]
  CPrint(3,'l.config.RcvPos=',l.config.RcvPos) #l.config.ContactBounce= 0.1
  for key,value in l.opt_conf['config'].iteritems():
    setattr(l.config, key, value)
  l.config.RcvSize= l._RcvSize

  m_setup.SetMaterial(l, preset=l._mtr_type) #('bounce','nobounce','natto','ketchup')
  l.config.SrcSize2H= l._SrcSize2H  #(0.02,0.09)
  CPrint(3,'l.config.ViscosityParam1=',l.config.ViscosityParam1)
  CPrint(3,'l.config.SrcSize2H=',l.config.SrcSize2H)

def Run(ct,*args):
  skill = args[0]   #("std_pour","shake_A","choose")
  mtr = args[1] if len(args)>=2 else None
  sms = args[2] if len(args)>=3 else None
  n_episode = args[3] if len(args)>=4 else 10
  
  target_logdir_name = "mtr_sms/infer/additional_more"
  model_dir_name = "mtr_sms/learn/additional_more"
  root_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  root_modeldir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  model_dir = root_modeldir + model_dir_name +"/models/"
  base_logdir = root_logdir + target_logdir_name + "/"

  reward_func = [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
                  TLocalQuad(3,lambda y:- 100.0*max(0.0,y[1]-y[0])**2 
                                        - 1.0*max(0.0,y[0]-y[1])**2 
                                        - 1.0*max(0.0,y[2])**2)]

  SP= TCompSpaceDef
  pour_skills = {
    'dtheta1': SP('action',1,min=[0.01],max=[0.02]),  #default min=[0.01],max=[0.02]
    'dtheta2': SP('action',1,min=[0.002],max=[0.005]),  #default min=[0.002],max=[0.005]
    'shake_spd': SP('action',1,min=[0.7],max=[0.9]),  #default min=[0.7],max=[0.9]
    'shake_axis2': SP('action',2,min=[0.05,-0.5*math.pi],max=[0.1,0.5*math.pi]),  #default min=[0.05,-0.5*math.pi],max=[0.1,0.5*math.pi]
  }
  
  opt_conf = {
    "skill_pour": skill, 
    'interactive': False,
    'not_learn': True,
    'num_episodes': n_episode,
    'num_log_interval': 1,
    'rwd_schedule': None,  #None, 'early_tip', 'early_shakeA'
    'model_dir': model_dir,  #'',
    'model_dir_persistent': False,  #If False, models are saved in l.logdir, i.e. different one from 'model_dir'
    'db_src': '',
    #'db_src': '/tmp/dpl/database.yaml',
    'config': {},  #Config of the simulator
    'dpl_options': {
      'opt_log_name': None,  #Not save optimization log.
      },
  }
  l= TContainer(debug=True)
  l._amount_trg= 0.3
  l._spilled_stop= 10
  l._RcvSize= [0.3, 0.4, 0.2]
  # l._mtr_type = "natto"      #('bounce','nobounce','natto','ketchup')
  # l._SrcSize2H= 0.02   #(0.02,0.09)

  if mtr=="bounce_list": mtr_list = ["bounce","nobounce"]
  elif mtr=="viscous_list": mtr_list = ["natto","ketchup"]
  elif mtr!=None: mtr_list = [mtr]
  else: mtr_list = ["bounce","nobounce","natto","ketchup"]
  if sms!=None: sms_list = [sms]
  else: sms_list = [0.02,0.055,0.09]

  for mtr in mtr_list:
    for sms in sms_list:
      set_name = mtr+"_"+str(sms).replace(".","")
      logdir = base_logdir + skill + "/" + set_name + "/"
      l._mtr_type = mtr
      l._SrcSize2H = sms

      ct.Run("mysim.try.onetime",
              l, 
              ConfigCallback,
              logdir, 
              opt_conf, 
              False, 
              reward_func, 
              pour_skills
              )

  # set_name = l._mtr_type+"_"+str(l._SrcSize2H).replace(".","")
  # logdir = base_logdir + skill + "/" + set_name + "/"
  # ct.Run("mysim.try_dpl_main",
  #             l, 
  #             ConfigCallback,
  #             logdir, 
  #             opt_conf, 
  #             False
  #             )