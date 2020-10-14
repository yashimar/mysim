import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pickle
from core_tool import *

def Help():  
  return '''Visualize dpl logs.
  Usage: mysim.vis_log'''

def Plot(df, ylabel,label, xmin=0, ymin=-30):
  for i in range(len(df.columns)):
    plt.plot(df.iloc[:,i], label=label)
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    plt.xlim(xmin,len(df))
    plt.ylim(ymin,0)
    plt.legend()
    plt.grid()
  plt.show()

def MtrScatter(df, mtr_list, vis_mtr_list, xmin=0, ymin=-30):
  c_dict = {"bounce":"purple","nobounce":"green","natto":"orange","ketchup":"red"}
  for mtr in list(set(mtr_list)):
    if mtr in vis_mtr_list:
      mtr_ids = [i for i, x in enumerate(mtr_list) if x==mtr]
      plt.scatter(mtr_ids, df.iloc[mtr_ids], label=mtr, c=c_dict[mtr])
  plt.xlim(xmin,len(mtr_list))
  plt.ylim(ymin,0)
  plt.legend()
  plt.show()

def PlotMeanStd(df, df_error, ylabel):
  plt.errorbar(df.index, df, df_error)
  plt.xlabel("episode")
  plt.ylabel(ylabel)
  plt.ylim(-30,0)
  plt.legend(loc='lower right')
  plt.grid()
  plt.show()

def Run(ct, *args):
  fig_name = "base50 + normal300"
  name_log_list = [
    "mtr_sms_sv/learning_branch/ranodm_mtr/base/shake_A/random/0055", 
    "mtr_sms_sv/learning_branch/ranodm_mtr/normal/shake_A/random/0055"
  ]
  r_thr = -1.0
  df_meta = pd.DataFrame()
  df_est_n0_meta = pd.DataFrame()
  df_est_last_meta = pd.DataFrame()
  mtr_list_meta = []
  smsz_list_meta = []

  for name_log in name_log_list:
    data_list = []
    mtr_list = []
    smsz_list = []
    root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
    log_path = root_path + name_log + "/dpl_est.dat"
    # db_path = root_path + name_log + "/sequence_list.yaml"
    config_path = root_path + name_log + "/config_log.yaml"
    # data_list.append(np.genfromtxt(log_path))
    data_list = []
    with open(log_path, "r") as log_data:
      for line in log_data:
        line = line.split("\n")[0].split(" ")
        line = map(lambda x: float(x), line)
        data_list.append(line)
    with open(config_path, "r") as yml:
      config = yaml.load(yml)
    for i in range(len(config)):
      conf = config[i]
      if conf["ContactBounce"]==0.7: mtr_list.append("bounce")
      elif conf["ViscosityParam1"]==1.5e-06: mtr_list.append("natto")
      elif conf["ViscosityParam1"]==2.5e-07: mtr_list.append("ketchup")
      else: mtr_list.append("nobounce")
      smsz_list.append(conf["SrcSize2H"])
    learn_ep_list = map(lambda x: x[0], data_list)
    mtr_list = [mtr for j,mtr in enumerate(mtr_list) if j in learn_ep_list]
    smsz_list = [smsz for j,smsz in enumerate(smsz_list) if j in learn_ep_list]
    learn_ep_before_sampling_list = [i for i in range(len(learn_ep_list)-1) 
                                    if learn_ep_list[i+1]-learn_ep_list[i]>=2]

    window = 10
    df_list = []
    df_est_n0_list = []
    df_est_last_list = []
    df_ma_list = []
    for i, data in enumerate(data_list):
      df_list.append(data[1])
      df_est_n0_list.append(data[2])
      df_est_last_list.append(data[-1])
    df = pd.DataFrame(df_list)
    df_est_n0 = pd.DataFrame(df_est_n0_list)
    df_est_last = pd.DataFrame(df_est_last_list)

    df_meta = pd.concat([df_meta, df])
    df_est_n0_meta = pd.concat([df_est_n0_meta, df_est_n0])
    df_est_last_meta = pd.concat([df_est_last_meta, df_est_last])
    mtr_list_meta += mtr_list
    smsz_list_meta += smsz_list

  df = df_meta.reset_index(drop=True)
  df_est_n0 = df_est_n0_meta.reset_index(drop=True)
  df_est_last = df_est_last_meta.reset_index(drop=True)
  mtr_list = mtr_list_meta
  smsz_list = smsz_list_meta
  
  plt.figure()
  plt.close()
  fig = plt.figure(figsize=(20,5))
  plt.title(fig_name) 
  
  vis_mtr_list = [
    "bounce", 
    "nobounce", 
    "natto", 
    "ketchup"
  ]
  xmin = 0
  ymin = -30
  # border = -0.5
  # plt.plot([border]*len(df), linestyle="dashed", alpha=0.5, c="red", label="border ("+str(border)+")")

  Plot(df.iloc[:,:], "return", "return", xmin, ymin)
  MtrScatter(df.iloc[:,:], mtr_list, vis_mtr_list, xmin, ymin)
  
  Plot(df_est_n0.iloc[:,:], "return", "estimate_n0", xmin , ymin)
  # MtrScatter(df_est_n0.iloc[:,:], mtr_list, vis_mtr_list, xmin, ymin)

  # Plot(df_est_last.iloc[:,:], "return", "estimate_last", xmin , ymin)
  # MtrScatter(df_est_last.iloc[:,:], mtr_list, vis_mtr_list, xmin, ymin)

  for ep in learn_ep_before_sampling_list:
    plt.text(ep,df.iloc[ep]-2,str(round(smsz_list[ep],3)))
  plt.vlines(learn_ep_before_sampling_list,ymin,0,
              linestyles="dashed",colors="gray",
              label="sampling timing")
  plt.legend()
  plt.subplots_adjust(left=0.05, right=0.95)
  # PlotMeanStd(df_ma["mean"], df_ma["std"], "reward_ma" + str(window))

  # plt.savefig("/home/yashima/Pictures/dpl3_7_21/"+name_log+".png")
  # plt.close()