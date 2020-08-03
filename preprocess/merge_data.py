#!/usr/bin/python
from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
import pickle
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
  print(l.logdir+"/"+dynamics+"_training_data_reference.txt")
  f = open(l.logdir+"/"+dynamics+"_training_data.txt", mode='w')
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
  with open(l.logdir+"/"+dynamics+"_training_data.pickle", mode='w') as fp:
    pickle.dump(dataX_merged,fp)
  f.close()

  return dataX_merged, dataY_merged

def Execute(l):
  dynamics_list = l.dynamics_list
  for dynamics in dynamics_list:
    CPrint(2,"===== ",dynamics,"=====")
    DataX, DataY = MergeData(l,dynamics)
    CPrint(3,"DataX shape:",DataX.shape)
    CPrint(3,"DataY shape:",DataY.shape)

def Run(ct,*args):
  log_dirname = "mtr_sms/merged_data"
  root_logpath = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/data/'
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
    "mtr_sms/learn/basic", 
    "random_sampled/mtr_sms/sample1", 
    "random_sampled/mtr_sms/sample2",
    "random_sampled/mtr_sms/sample3",
    "random_sampled/mtr_sms/sample4",
  ]
  data_dir_list = map(lambda x: root_modeldir+x+"/models/", data_dir_list)

  dynamics_list = ['Fgrasp','Fmvtorcv_rcvmv','Fmvtorcv','Fmvtopour2',
                    'Fflowc_tip10','Fflowc_shakeA10','Famount4']

  l= TContainer(debug=True)
  l.data_dir_list = data_dir_list
  l.dynamics_list = dynamics_list
  l.logdir = root_logpath + log_dirname

  Execute(l)

  l= None
  return True
