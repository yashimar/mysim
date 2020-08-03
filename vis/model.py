import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from core_tool import *

def Help():  
  return '''Visualize model logs.
  Usage: mysim.vis_model'''

def Run(ct, *args):
  log_name = args[0]
  model_logs = defaultdict()
  # dynamics_list = ["Fgrasp","Fmvtorcv_rcvmv","Fmvtorcv","Fmvtopour2"]
  # dynamics_list = ["Fflowc_tip10","Fflowc_shakeA10","Famount4"]
  dynamics_list = ["Fgrasp","Fmvtorcv_rcvmv","Fmvtorcv","Fmvtopour2",
                    "Fflowc_tip10","Fflowc_shakeA10","Famount4"]
  fig = plt.figure(figsize=(5*len(dynamics_list),5))
  fig.suptitle(log_name, fontsize=12)
  for count, dynamics in enumerate(dynamics_list):
    mean_list = []
    err_list = []
    i = 0;
    while i<1000:
      # base_path = "/tmp/learn_grab/models/train/"
      # base_path = "/tmp/dpl01/models/train/"
      # base_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/learn_dynamics/models/train/"
      base_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" + log_name + "/models/train/"
      name = "nn_log-" + "0"*(5-len(str(i)))+str(i) + "-" + dynamics
      mean_path = base_path + name + "mean" + ".dat"
      err_path = base_path + name + "err" + ".dat"
      if os.path.exists(mean_path):
        mean_list.append(np.loadtxt(mean_path, comments='!')[-1][2])
      if os.path.exists(err_path):  
        err_list.append(np.loadtxt(err_path, comments='!')[-1][2])
      i += 1
    model_logs[dynamics+"_mean"] = mean_list
    model_logs[dynamics+"_err"] = err_list

    ax = fig.add_subplot(2,len(dynamics_list),count+1)
    plt.subplots_adjust(wspace=0.7, hspace=0.6)
    ax.set_ylim(1e-5,0.01)
    ax.set_title(dynamics+" mean model",fontsize=9)
    ax.set_xlabel("episode",fontsize=9)
    ax.set_ylabel("ema loss",fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.plot(mean_list)

    ax = fig.add_subplot(2,len(dynamics_list),count+len(dynamics_list)+1)
    plt.subplots_adjust(wspace=0.7, hspace=0.6)
    ax.set_ylim(1e-5,0.003)
    ax.set_title(dynamics+" error model",fontsize=9)
    ax.set_xlabel("episode",fontsize=9)
    ax.set_ylabel("ema loss",fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.plot(err_list)

  plt.plot()