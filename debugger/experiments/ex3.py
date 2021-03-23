from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
from collections import defaultdict

PLANNING = "planning"
SAMPLING = "sampling"

def Help():
  pass

def ConfigCallback(ct,l,sim):
  m_setup= ct.Load('mysim.setup.setup_sv')
  l.amount_trg= 0.3
  l.spilled_stop= 10
  l.config.RcvPos= [0.6, l.config.RcvPos[1], l.config.RcvPos[2]]
  # l.config.RcvPos= [0.8+0.6*(random.random()-0.5), l.config.RcvPos[1], l.config.RcvPos[2]]
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
      l.latest_mtr = ('bounce','nobounce','natto','ketchup')[RandI(4)]
      m_setup.SetMaterial(l, preset=l.latest_mtr)
    else:
      l.latest_mtr = l.custom_mtr
      m_setup.SetMaterial(l, preset=l.custom_mtr)
    if l.custom_smsz=="random":
      l.config.SrcSize2H= Rand(0.03,0.08)
    else:
      l.config.SrcSize2H= l.custom_smsz
    l.latest_smsz = l.config.SrcSize2H
  elif l.mtr_smsz=="latest_mtr_smsz":
    m_setup.SetMaterial(l, preset=l.latest_mtr)
    l.config.SrcSize2H= Rand(max(0.03,l.latest_smsz-l.delta_smsz), 
                              min(0.08,l.latest_smsz+l.delta_smsz))
  elif l.mtr_smsz=="early_natto":
    m_setup.SetMaterial(l, preset='natto')
    l.config.SrcSize2H = l.custom_smsz
  CPrint(3,'l.config.ViscosityParam1=',l.config.ViscosityParam1)
  CPrint(3,'l.config.SrcSize2H=',l.config.SrcSize2H)

def Run(ct,*args):
  run_type = args[0] #LEANRING, SAMPLING
  sa_dict = args[1] if len(args)==3 else None
  save_path = args[2] if len(args)==3 else None

  l = TContainer(debug=True)
  l.run_type = run_type
  l.logdir= '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
            + "debugger/ex3"+"/"
  # l.logdir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
  #             + "mtr_sms_sv/test/learning_branch/"
  # l.logdir = "/tmp/lb/"
  suff = "first"+"/"
  
  l.config_callback= ConfigCallback
  l.type = "dnn"

  if l.run_type==PLANNING:
    # src_core = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
    #         + "bottomup/learn4/std_pour/nobounce/random/graphModel/modifiedStdPour/first"+"/"
    # model_dir = src_core + "models/"
    # db_src = src_core + "database.yaml"
    model_dir = ""
    src_core = ""
    db_src = ""

    l.pour_skill = "std_pour"
    l.custom_mtr = "nobounce"
    # l.custom_smsz = 0.065    #random or 0.03~0.08
    l.custom_smsz = "random"    #random or 0.03~0.08
    l.delta_smsz = 0.0
    l.mtr_dir_name = "nobounce"

    l.opt_conf={
      'interactive': False,
      'not_learn': False,
      'num_episodes': 200,
      'max_priority_sampling': 0, 
      # "sampling_mode": "random", #random, bo(bayesian optimization)
      "return_epsiron": -100.0, 
      'num_log_interval': 1,  #should be 1
      'rcv_size': 'static',  #'static', 'random'
      'mtr_smsz': 'custom',  #'fixed', 'fxvs1', 'random', 'viscous', custom
      "planning_node": ["n0"], #"n0","n2a"
      'rwd_schedule': None,  #None, 'early_tip', 'early_shakeA', "only_tip", "only_shakeA"
      'mtr_schedule': None,  #None, "early_natto"
      'model_dir': model_dir,
      'model_dir_persistent': False,  #If False, models are saved in l.logdir, i.e. different one from 'model_dir'
      'db_src': db_src,
      'config': {},  #Config of the simulator
      'dpl_options': {
        'opt_log_name': '{base}seq/opt-{i:04d}-{e:03d}-{n}-{v:03d}.dat',  #'{base}seq/opt-{i:04d}-{e:03d}-{n}-{v:03d}.dat' or None
        "ddp_sol":{
            # 'ptree_num': 40, #default auto
            # 'db_init_ratio': 1.0, #default 0.5
            'db_init_R_min': -1.0, #default -1.0
            'grad_max_bounce': 10, #default 10
            'prob_update_best': 0.4, #default 0.4
            'prob_update_rand': 0.3, #default 0.3
            'max_total_iter': 2000, #default 2000 
            "grad_max_iter": 50,  #default 50
            'gd_alpha': 0.03 #default 0.03
          },
        },
      }
    l.nn_options = {
      # "gpu": 0, 
      "batchsize": 10,           #default 10
      "num_max_update": 5000,     #default 5000
      'num_check_stop': 50,       #default 50
      'loss_stddev_stop': 1e-3,  #default 1e-3
      'AdaDelta_rho': 0.9,        #default 0.9
      # 'train_log_file': '{base}train/nn_log-{name}{code}.dat', 
      # "train_batch_loss_log_file": '{base}train/nn_batch_loss_log-{name}{code}.dat',
    }
    if l.custom_smsz=="random": smsz_label="random"
    else: smsz_label = str(l.custom_smsz).split(".")[0] + str(l.custom_smsz).split(".")[1]
    l.logdir = l.logdir + l.pour_skill + "/" + l.mtr_dir_name + "/" + smsz_label + "/" + suff
    print 'Copying',PycToPy(__file__),'to',PycToPy(l.logdir+os.path.basename(__file__))
    CopyFile(PycToPy(__file__),PycToPy(l.logdir+os.path.basename(__file__)))
    if os.path.exists(l.logdir+"config_log.yaml")==False and os.path.exists(l.logdir+"dpl_est.dat")==False:
      if src_core!="":
        CopyFile(src_core+"config_log.yaml", l.logdir+"config_log.yaml")
        Print("Copying",src_core+"config_log.yaml","to",l.logdir+"config_log.yaml")
        CopyFile(src_core+"dpl_est.dat", l.logdir+"dpl_est.dat")
        Print("Copying",src_core+"dpl_est.dat","to",l.logdir+"dpl_est.dat")
      else:
        os.mknod(l.logdir+"config_log.yaml")
        os.mknod(l.logdir+"dpl_est.dat")
    else:
      pass

  elif l.run_type==SAMPLING:
    model_dir = ""
    src_core = ""
    db_src = ""

    l.pour_skill = "std_pour"
    l.custom_mtr = "nobounce"
    l.custom_smsz_all = sa_dict["size_srcmouth"]
    n_episode = len(l.custom_smsz_all)

    l.skill_params = {
      'gh_ratio': [SSA([0.5])]*n_episode,
      'p_pour_trg0': lambda pc_rcv: [SSA(Vec([-0.3,0.35])+Vec([pc_rcv[0],pc_rcv[2]]))]*n_episode,
      'p_pour_trg': [SSA(x) for x in sa_dict["p_pour_trg"]],
      # 'p_pour_trg': lambda pc_rcv: [SSA(Vec([x+0.6,0.31+0.202])) for x in np.linspace(-0.1, 0.0, 20)]*20,
      # 'p_pour_trg': lambda pc_rcv: [SSA(Vec([-0.1+0.6,0.15+0.202]))]*n_episode,
      'dtheta1': [SSA([0.014])]*n_episode,
      # 'dtheta2': [SSA([0.004])]*n_episode,
      'dtheta2': [SSA([x]) for x in sa_dict["dtheta2"]],
      'shake_spd': [SSA([0.8])]*n_episode,
      'shake_axis2': [SSA([0.08,0.0])]*n_episode
    }

    l.opt_conf={
      'interactive': False,
      'not_learn': True,
      'num_episodes': n_episode,
      'max_priority_sampling': 0, 
      # "sampling_mode": "random", #random, bo(bayesian optimization)
      "return_epsiron": -100.0, 
      'num_log_interval': 1,  #should be 1
      'rcv_size': 'static',  #'static', 'random'
      'mtr_smsz': 'custom',  #'fixed', 'fxvs1', 'random', 'viscous', custom
      "planning_node": ["n0"], #"n0","n2a"
      'rwd_schedule': None,  #None, 'early_tip', 'early_shakeA', "only_tip", "only_shakeA"
      'mtr_schedule': None,  #None, "early_natto"
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
    l.nn_options = {
      # # "gpu": 0, 
      # "batchsize": 10,           #default 10
      # "num_max_update": 5000,     #default 5000
      # 'num_check_stop': 50,       #default 50
      # 'loss_stddev_stop': 1e-3,  #default 1e-3
      # 'AdaDelta_rho': 0.9,        #default 0.9
      # # 'train_log_file': '{base}train/nn_log-{name}{code}.dat', 
      # # "train_batch_loss_log_file": '{base}train/nn_batch_loss_log-{name}{code}.dat',
    }
    # if l.custom_smsz=="random": smsz_label="random"
    # else: smsz_label = str(l.custom_smsz).split(".")[0] + str(l.custom_smsz).split(".")[1]
    smsz_label = "custom"
    l.logdir = save_path
    print 'Copying',PycToPy(__file__),'to',PycToPy(l.logdir+os.path.basename(__file__))
    # CopyFile(PycToPy(__file__),PycToPy(l.logdir+os.path.basename(__file__)))
    if not os.path.isdir(l.logdir):
      os.makedirs(l.logdir)
    if os.path.exists(l.logdir+"config_log.yaml")==False and os.path.exists(l.logdir+"dpl_est.dat")==False:
      if src_core!="":
        CopyFile(src_core+"config_log.yaml", l.logdir+"config_log.yaml")
        Print("Copying",src_core+"config_log.yaml","to",l.logdir+"config_log.yaml")
        CopyFile(src_core+"dpl_est.dat", l.logdir+"dpl_est.dat")
        Print("Copying",src_core+"dpl_est.dat","to",l.logdir+"dpl_est.dat")
      else:
        os.mknod(l.logdir+"config_log.yaml")
        os.mknod(l.logdir+"dpl_est.dat")
    else:
      pass

  ct.Run("mysim.debugger.experiments.ex3_main", l)