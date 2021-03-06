from core_tool import *

def Help():
  pass

def merge_dicts(a, b):
  m = a.copy()
  m.update(b)
  return m

def Run(ct,*args):
  do_update = True
  target_logdir = 'mtr_sms/learn/additional3_early'
  # target_logdir = "test_batch_loss_log"
  base_modeldir = "mtr_sms/learn/basic2"
  root_logdir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'
  # root_logdir = "/tmp/"
  logdir= root_logdir + target_logdir+"/"
  # root_modeldir = root_dir
  root_modeldir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'
  data_dir_list = [
    # "dpl01", 
    # "dpl2_std_pour11311", 
    # "dpl2_std_pour11312", 
    # "dpl2_std_pour11313", 
    # "dpl2_std_pour11314", 
    # "dpl2_std_pour11315", 
    # "dpl2_shake_A11311", 
    # "dpl2_shake_A11312", 
    # "dpl2_shake_A11313", 
    # "dpl2_shake_A11314", 
    # "dpl2_shake_A11315", 
    # "dpl2_choose_skill11311", 
    # "dpl2_choose_skill11312", 
    # "dpl2_choose_skill11313", 
    # "dpl2_choose_skill11314", 
    # "dpl2_choose_skill11315", 
    # "dpl3_std_pour11312", 
    # "dpl3_shake_A11311", 
    # "dpl3_choose_skill11313", 
    # "learn_dynamics_dpl3", 

    # "gather_sample/viscous1",
    # "gather_sample/viscous2",
    # "gather_sample/viscous3",
    # "gather_sample/viscous4",

    "mtr_sms/learn/basic2",

    "random_sampled/mtr_sms/sample1", 
    "random_sampled/mtr_sms/sample2",
    "random_sampled/mtr_sms/sample3",
    "random_sampled/mtr_sms/sample4",
  ]
  data_dir_list = map(lambda x: root_modeldir+x+"/models/", data_dir_list)
  opt_conf={
    'interactive': False,
    'not_learn': False,
    "model_dir": root_modeldir + base_modeldir + "/models/", 
    # "model_dir": "", 
    'model_dir_persistent': False,
    # "db_src": root_dir + "dpl01/database.yaml",
    "db_src": "", 
    'config': {},  #Config of the simulator
    'dpl_options': {
      'opt_log_name': None,  #Not save optimization log.
      },
    }
  nn_options_base = {
    # "gpu": 0, 
    "batch_size": 10,           #default 10
    "num_max_update": 100000,     #default 5000
    'num_check_stop': 50,       #default 50
    'loss_stddev_stop': 1e-3,  #default 1e-3
    'AdaDelta_rho': 0.9,        #default 0.9
    'train_log_file': '{base}train/nn_log-{name}{code}.dat', 
    "train_batch_loss_log_file": '{base}train/nn_batch_loss_log-{name}{code}.dat',
  }
  nn_options = {
    "Fgrasp": merge_dicts(nn_options_base,{
      # 'num_check_stop': 250
      }), 
    "Fmvtorcv_rcvmv": merge_dicts(nn_options_base, {
      # 'num_check_stop': 250
      }), 
    "Fmvtorcv": merge_dicts(nn_options_base, {
      # 'num_check_stop': 250
      }), 
    "Fmvtopour2": merge_dicts(nn_options_base, {
      # 'num_check_stop': 500
      }), 
    "Fflowc_tip10": merge_dicts(nn_options_base, {
      # 'num_check_stop': 300
      }), 
    "Fflowc_shakeA10": merge_dicts(nn_options_base, {
      # 'num_check_stop': 300
      }), 
    "Famount4": merge_dicts(nn_options_base, {
      # 'num_check_stop': 50, 
      # "loss_stddev_stop": 1e-3, 
      })
  }
  # dynamics_list = ['Fgrasp','Fmvtorcv_rcvmv','Fmvtorcv','Fmvtopour2']
  # dynamics_list = ['Fflowc_tip10','Fflowc_shakeA10','Famount4']
  dynamics_list = ['Fgrasp','Fmvtorcv_rcvmv','Fmvtorcv','Fmvtopour2',
                    'Fflowc_tip10','Fflowc_shakeA10','Famount4']
  # dynamics_list = ["Famount4"]

  print 'Copying',PycToPy(__file__),'to',PycToPy(logdir+os.path.basename(__file__))
  CopyFile(PycToPy(__file__),PycToPy(logdir+os.path.basename(__file__)))

  ct.Run(
    "mysim.learn.learning_main", 
    logdir, 
    opt_conf, 
    nn_options, 
    data_dir_list, 
    dynamics_list, 
    do_update
  )