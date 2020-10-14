import numpy as np
from matplotlib import pyplot as plt
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  log_name = args[0]
  is_onpolicy = args[1] if len(args)==2 else True
  # dynamics_list = ["Fgrasp"]
  dynamics_list = [
    # "Fgrasp","Fmvtorcv_rcvmv","Fmvtorcv","Fmvtopour2",
    # "Fflowc_tip10",
    # "Fflowc_shakeA10",
    "Famount4"
    ]

  data_list = []
  log_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" + log_name + "/dpl_est.dat"
  with open(log_path, "r") as log_data:
    for line in log_data:
      line = line.split("\n")[0].split(" ")
      line = map(lambda x: float(x), line)
      data_list.append(line)
  learn_ep_list = map(lambda x: x[0], data_list)
  learn_ep_before_sampling_list = [i for i in range(len(learn_ep_list)-1) 
                                  if learn_ep_list[i+1]-learn_ep_list[i]>=2]
  
  for count,dynamics in enumerate(dynamics_list):
    mean_batch_loss_list = []
    median_mean_batch_loss_list = []
    q3_mean_batch_loss_list = []
    q1_mean_batch_loss_list = []
    median_mean_batch_loss_list_before_sampling = []
    q3_mean_batch_loss_list_before_sampling = []
    q1_mean_batch_loss_list_before_sampling = []
    err_batch_loss_list = []
    mean_batch_start_loss_list = []
    err_batch_start_loss_list = []
    trial_mean_batch_dict = {}
    if is_onpolicy:
      for i in range(1000):
        base_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" + log_name + "/models/train/"
        name = "nn_batch_loss_log-" + "0"*(5-len(str(i)))+str(i) + "-" + dynamics
        mean_path = base_path + name + "mean" + ".dat"
        err_path = base_path + name + "err" + ".dat"
        if os.path.exists(mean_path):
          loss_list = list(np.loadtxt(mean_path, comments='!').transpose()[1])
          median_mean_batch_loss_list += [np.median(loss_list)]*len(loss_list)
          q3_mean_batch_loss_list += [np.percentile(loss_list,75)]*len(loss_list)
          q1_mean_batch_loss_list += [np.percentile(loss_list,25)]*len(loss_list)
          if i not in learn_ep_before_sampling_list: 
            median_mean_batch_loss_list_before_sampling += [np.nan]*len(loss_list)
            q3_mean_batch_loss_list_before_sampling += [np.nan]*len(loss_list)
            q1_mean_batch_loss_list_before_sampling += [np.nan]*len(loss_list)
          else:
            median_mean_batch_loss_list_before_sampling += [np.median(loss_list)]*len(loss_list)
            q3_mean_batch_loss_list_before_sampling += [np.percentile(loss_list,75)]*len(loss_list)
            q1_mean_batch_loss_list_before_sampling += [np.percentile(loss_list,25)]*len(loss_list)
          mean_batch_start_loss_list += [loss_list[0]] + [np.nan]*(len(loss_list)-1)
          mean_batch_loss_list += loss_list
          final_mean_batch_loss = loss_list
          trial_mean_batch_dict.update({i: len(mean_batch_loss_list)-len(loss_list)})
        if os.path.exists(err_path):  
          loss_list = list(np.loadtxt(err_path, comments='!').transpose()[1])
          err_batch_start_loss_list += [loss_list[0]] + [np.nan]*(len(loss_list)-1)
          err_batch_loss_list += loss_list
          final_err_batch_loss = loss_list
    else:
      base_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" + log_name + "/models/train/"
      name = "nn_batch_loss_log-" + dynamics
      mean_path = base_path + name + "mean" + ".dat"
      err_path = base_path + name + "err" + ".dat"
      mean_batch_loss_list += list(np.loadtxt(mean_path, comments='!').transpose()[1])
      err_batch_loss_list += list(np.loadtxt(err_path, comments='!').transpose()[1])

    if not dynamics=="Famount4": continue
    plt.figure()
    plt.close()

    plt.figure(figsize=(20,10))
    plt.title(log_name+"\n"+dynamics+" mean model")
    plt.xlabel("update")
    plt.ylabel("batch loss")
    plt.ylim(0,0.025)
    # plt.plot(final_mean_batch_loss, linewidth=0.8, zorder=0)
    plt.xlim(200000,220000)
    plt.plot(mean_batch_loss_list, linewidth=0.1, zorder=0)
    plt.plot(median_mean_batch_loss_list, zorder=2)
    plt.plot(median_mean_batch_loss_list_before_sampling, zorder=3, linewidth=2, c="black")
    plt.plot(q3_mean_batch_loss_list, zorder=2, c="purple")
    plt.plot(q3_mean_batch_loss_list_before_sampling, zorder=3, linewidth=2, c="black")
    plt.plot(q1_mean_batch_loss_list, zorder=2, c="pink")
    plt.plot(q1_mean_batch_loss_list_before_sampling, zorder=3, linewidth=2, c="black")
    plt.scatter(np.linspace(0,len(mean_batch_start_loss_list)-1,len(mean_batch_start_loss_list)),mean_batch_start_loss_list, c="red", zorder=1)
    # plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()
    # print(trial_mean_batch_dict)
    # print([i for i,loss in enumerate(mean_batch_loss_list) if loss>1.5])

    # if is_onpolicy: fig = plt.figure(figsize=(5*len(dynamics_list),10))
    # else: fig = plt.figure(figsize=(5*len(dynamics_list),5))
    # fig.suptitle(log_name, fontsize=12)

    # if is_onpolicy: rows=4
    # else: rows=2

    # ax = fig.add_subplot(rows,len(dynamics_list),count+1)
    # plt.subplots_adjust(wspace=0.7, hspace=0.6)
    # if dynamics=="Famount4": ax.set_ylim(1e-5,0.5)
    # else: ax.set_ylim(1e-5,0.05)
    # ax.set_title(dynamics+" mean model",fontsize=9)
    # ax.set_xlabel("update",fontsize=9)
    # ax.set_ylabel("batch loss",fontsize=9)
    # ax.tick_params(axis='x', labelsize=8)
    # ax.tick_params(axis='y', labelsize=8)
    # ax.scatter(np.linspace(0,len(mean_batch_start_loss_list)-1,len(mean_batch_start_loss_list)),mean_batch_start_loss_list, c="red")
    # if dynamics=="Famount4": ax.plot(mean_batch_loss_list, linewidth=0.06)
    # else: ax.plot(mean_batch_loss_list, linewidth=0.06)
    

    # ax = fig.add_subplot(rows,len(dynamics_list),count+(rows/2)*len(dynamics_list)+1)
    # plt.subplots_adjust(wspace=0.7, hspace=0.6)
    # if dynamics=="Famount4": ax.set_ylim(1e-5,0.05)
    # else: ax.set_ylim(1e-5,0.005)
    # ax.set_title(dynamics+" error model",fontsize=9)
    # ax.set_xlabel("update",fontsize=9)
    # ax.set_ylabel("batch loss",fontsize=9)
    # ax.tick_params(axis='x', labelsize=8)
    # ax.tick_params(axis='y', labelsize=8)
    # ax.scatter(np.linspace(0,len(err_batch_start_loss_list)-1,len(err_batch_start_loss_list)),err_batch_start_loss_list, c="red")
    # ax.plot(err_batch_loss_list,c="orange", linewidth=0.8)

    # if is_onpolicy:
    #   ax = fig.add_subplot(rows,len(dynamics_list),len(dynamics_list)+count+1)
    #   plt.subplots_adjust(wspace=0.7, hspace=0.6)
    #   if dynamics=="Famount4": ax.set_ylim(1e-5,0.05)
    #   else: ax.set_ylim(1e-5,0.05)
    #   ax.set_title(dynamics+" mean model\n(final episode)",fontsize=9)
    #   ax.set_xlabel("update",fontsize=9)
    #   ax.set_ylabel("batch loss",fontsize=9)
    #   ax.tick_params(axis='x', labelsize=8)
    #   ax.tick_params(axis='y', labelsize=8)
    #   ax.plot(final_mean_batch_loss, linewidth=0.8)

    #   ax = fig.add_subplot(rows,len(dynamics_list),count+3*len(dynamics_list)+1)
    #   plt.subplots_adjust(wspace=0.7, hspace=0.6)
    #   if dynamics=="Famount4": ax.set_ylim(1e-5,0.005)
    #   else: ax.set_ylim(1e-5,0.005)
    #   ax.set_title(dynamics+" error model\n(final episode)",fontsize=9)
    #   ax.set_xlabel("update",fontsize=9)
    #   ax.set_ylabel("batch loss",fontsize=9)
    #   ax.tick_params(axis='x', labelsize=8)
    #   ax.tick_params(axis='y', labelsize=8)
    #   ax.plot(final_err_batch_loss,c="orange", linewidth=0.8)

  plt.show()
  # fig.savefig("/home/yashima/Pictures/mtr_sms/learning_curve/"+args[0].split("/")[-1]+".png")