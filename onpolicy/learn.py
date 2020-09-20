from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
from collections import defaultdict

def Help():
  pass

def ConfigCallback(ct,l,sim):
  m_setup= ct.Load('mysim.setup.setup_sv')
  l.amount_trg= 0.3
  l.spilled_stop= 10
  l.config.RcvPos= [0.6, l.config.RcvPos[1], l.config.RcvPos[2]]
  CPrint(3,'l.config.RcvPos=',l.config.RcvPos)
  for key,value in l.opt_conf['config'].iteritems():
    setattr(l.config, key, value)

  if l.rcv_size=='static':
    l.config.RcvSize= [0.3, 0.4, 0.2]
  elif l.rcv_size=='random':
    rsx= Rand(0.25,0.5)
    rsy= Rand(0.1,0.2)/rsx
    rsz= Rand(0.2,0.5)
    l.config.RcvSize= [rsx, rsy, rsz]

  if l.mtr_smsz=='fixed':
    m_setup.SetMaterial(l, preset='bounce')
    l.config.SrcSize2H= 0.03  #Mouth size of source container
  elif l.mtr_smsz=='fxvs1':
    m_setup.SetMaterial(l, preset='ketchup')
    l.config.SrcSize2H= 0.08  #Mouth size of source container
  elif l.mtr_smsz=='random':
    m_setup.SetMaterial(l, preset=('bounce','nobounce','natto','ketchup')[RandI(4)])
    l.config.SrcSize2H= Rand(0.02,0.09)  #Mouth size of source container
  elif l.mtr_smsz=='viscous':
    m_setup.SetMaterial(l, preset=('natto','ketchup')[RandI(2)])
    l.config.SrcSize2H= Rand(0.05,0.09)  #Mouth size of source container
  elif l.mtr_smsz=="custom":
    if l.custom_mtr=="random":
      m_setup.SetMaterial(l, preset=('bounce','nobounce','natto','ketchup')[RandI(4)])
    else:
      m_setup.SetMaterial(l, preset=l.custom_mtr)
    l.config.SrcSize2H= l.custom_smsz
  CPrint(3,'l.config.ViscosityParam1=',l.config.ViscosityParam1)
  CPrint(3,'l.config.SrcSize2H=',l.config.SrcSize2H)

def Run(ct,*args):
  l = TContainer(debug=True)
  l.logdir= '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
            + "mtr_sms_sv/learn/"
  # suff = "continuous"
  suff = "normal"
  # model_dir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
  #           + "mtr_sms_sv/learn/shake_A/random/0055/normal/models/"
  model_dir = ""
  # db_src = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
  #         + "mtr_sms_sv/learn/shake_A/random/0055/normal/database.yaml"
  db_src = ""
  l.pour_skill = "shake_A"

  l.config_callback= ConfigCallback
  l.custom_mtr = "random"
  l.custom_smsz = 0.055

  l.opt_conf={
    'interactive': False,
    'not_learn': False,
    'num_episodes': 190,
    'num_log_interval': 1,  #should be 1
    'rcv_size': 'static',  #'static', 'random'
    'mtr_smsz': 'custom',  #'fixed', 'fxvs1', 'random', 'viscous', custom
    'rwd_schedule': None,  #None, 'early_tip', 'early_shakeA', "only_tip", "only_shakeA"
    'model_dir': model_dir,
    'model_dir_persistent': False,  #If False, models are saved in l.logdir, i.e. different one from 'model_dir'
    'db_src': db_src,
    'config': {},  #Config of the simulator
    'dpl_options': {
      'opt_log_name': '{base}seq/opt-{i:04d}-{e:03d}-{n}-{v:03d}.dat',  #'{base}seq/opt-{i:04d}-{e:03d}-{n}-{v:03d}.dat' or None
      "ddp_sol":{
          'db_init_ratio': 0.5, #default 0.5
          'prob_update_best': 0.4, #default 0.4
          'prob_update_rand': 0.3, #default 0.3
          'max_total_iter': 2000, #default 2000 
          "grad_max_iter": 50,  #default 50
          'gd_alpha': 0.03 #default 0.03
        },
      },
    }

  c = str(l.custom_smsz).split(".")
  l.logdir = l.logdir + l.pour_skill + "/" + l.custom_mtr + "/" + c[0]+c[1] + "/" + suff + "/"
  print 'Copying',PycToPy(__file__),'to',PycToPy(l.logdir+os.path.basename(__file__))
  CopyFile(PycToPy(__file__),PycToPy(l.logdir+os.path.basename(__file__)))

  ct.Run("mysim.onpolicy.basic_sv", l)