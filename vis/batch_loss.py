import numpy as np
from matplotlib import pyplot as plt
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  log_name = args[0]
  is_onpolicy = args[1] if len(args)==2 else True
  # dynamics_list = ["Fgrasp"]
  dynamics_list = ["Fgrasp","Fmvtorcv_rcvmv","Fmvtorcv","Fmvtopour2",
                    "Fflowc_tip10","Fflowc_shakeA10","Famount4"]
  
  if is_onpolicy: fig = plt.figure(figsize=(5*len(dynamics_list),10))
  else: fig = plt.figure(figsize=(5*len(dynamics_list),5))
  fig.suptitle(log_name, fontsize=12)
  for count,dynamics in enumerate(dynamics_list):
    mean_batch_loss_list = []
    err_batch_loss_list = []
    if is_onpolicy:
      for i in range(1000):
        base_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" + log_name + "/models/train/"
        name = "nn_batch_loss_log-" + "0"*(5-len(str(i)))+str(i) + "-" + dynamics
        mean_path = base_path + name + "mean" + ".dat"
        err_path = base_path + name + "err" + ".dat"
        if os.path.exists(mean_path):
          mean_batch_loss_list += list(np.loadtxt(mean_path, comments='!').transpose()[1])
          final_mean_batch_loss = list(np.loadtxt(mean_path, comments='!').transpose()[1])
        if os.path.exists(err_path):  
          err_batch_loss_list += list(np.loadtxt(err_path, comments='!').transpose()[1])
          final_err_batch_loss = list(np.loadtxt(err_path, comments='!').transpose()[1])
    else:
      base_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" + log_name + "/models/train/"
      name = "nn_batch_loss_log-" + dynamics
      mean_path = base_path + name + "mean" + ".dat"
      err_path = base_path + name + "err" + ".dat"
      mean_batch_loss_list += list(np.loadtxt(mean_path, comments='!').transpose()[1])
      err_batch_loss_list += list(np.loadtxt(err_path, comments='!').transpose()[1])

    
    if is_onpolicy: rows=4
    else: rows=2

    ax = fig.add_subplot(rows,len(dynamics_list),count+1)
    plt.subplots_adjust(wspace=0.7, hspace=0.6)
    if dynamics=="Famount4": ax.set_ylim(1e-5,0.5)
    else: ax.set_ylim(1e-5,0.05)
    ax.set_title(dynamics+" mean model",fontsize=9)
    ax.set_xlabel("update",fontsize=9)
    ax.set_ylabel("batch loss",fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    if dynamics=="Famount4": ax.plot(mean_batch_loss_list, linewidth=0.06)
    else: ax.plot(mean_batch_loss_list, linewidth=0.06)
    

    ax = fig.add_subplot(rows,len(dynamics_list),count+(rows/2)*len(dynamics_list)+1)
    plt.subplots_adjust(wspace=0.7, hspace=0.6)
    if dynamics=="Famount4": ax.set_ylim(1e-5,0.05)
    else: ax.set_ylim(1e-5,0.005)
    ax.set_title(dynamics+" error model",fontsize=9)
    ax.set_xlabel("update",fontsize=9)
    ax.set_ylabel("batch loss",fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.plot(err_batch_loss_list,c="orange", linewidth=0.8)

    if is_onpolicy:
      ax = fig.add_subplot(rows,len(dynamics_list),len(dynamics_list)+count+1)
      plt.subplots_adjust(wspace=0.7, hspace=0.6)
      if dynamics=="Famount4": ax.set_ylim(1e-5,0.05)
      else: ax.set_ylim(1e-5,0.05)
      ax.set_title(dynamics+" mean model\n(final episode)",fontsize=9)
      ax.set_xlabel("update",fontsize=9)
      ax.set_ylabel("batch loss",fontsize=9)
      ax.tick_params(axis='x', labelsize=8)
      ax.tick_params(axis='y', labelsize=8)
      ax.plot(final_mean_batch_loss, linewidth=0.8)

      ax = fig.add_subplot(rows,len(dynamics_list),count+3*len(dynamics_list)+1)
      plt.subplots_adjust(wspace=0.7, hspace=0.6)
      if dynamics=="Famount4": ax.set_ylim(1e-5,0.005)
      else: ax.set_ylim(1e-5,0.005)
      ax.set_title(dynamics+" error model\n(final episode)",fontsize=9)
      ax.set_xlabel("update",fontsize=9)
      ax.set_ylabel("batch loss",fontsize=9)
      ax.tick_params(axis='x', labelsize=8)
      ax.tick_params(axis='y', labelsize=8)
      ax.plot(final_err_batch_loss,c="orange", linewidth=0.8)

  plt.show()
  fig.savefig("/home/yashima/Pictures/mtr_sms/learning_curve/"+args[0].split("/")[-1]+".png")