import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pickle
from core_tool import *

def Help():  
  return '''Visualize dpl logs.
  Usage: mysim.vis_log'''

def Plot(df, ylabel,ylim=-30,ymax=0):
  try:
    for i in range(len(df.columns)):
      plt.plot(df.iloc[:,i], label=df.columns[i])
      plt.xlabel("episode")
      plt.ylabel(ylabel)
      plt.ylim(ylim,ymax)
      plt.legend()
      plt.grid()
  except:
    plt.plot(df)
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    plt.ylim(ylim,ymax)
    plt.legend()
    plt.grid()
  plt.show()

def PlotMeanStd(ax, df, title):
  ax.errorbar(df.index, df["mean"], df["std"], ecolor="orange")
  ax.set_title(title)
  ax.set_xlabel("episode")
  ax.set_ylabel("mean reward")
  ax.set_ylim(-30,2)
  plt.legend(loc='lower right')
  ax.grid()
  # plt.show()

def Scatter(ax, df, ylabel):
  for i in range(len(df.columns)):
    ax.scatter(x=df.index, y=df.iloc[:,i], s=15, label=df.columns[i])
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-60)
    ax.grid()
    plt.legend()
  # plt.show()

def Run(ct, *args):
  mode = args[1]
  i = 0;
  data_list = []
  materials_list = []
  while True:
    try:
      log_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" \
                      + args[0] + str(11311+i) + "/dpl_est.dat"
      # database_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" \
      #                 + args[0] + str(11311+i) + "/database.yaml"
      data_list.append(np.loadtxt(log_path, comments='!'))
      # with open(database_path) as f:
      #   database = yaml.safe_load(f)
      # materials = []
      # for j in range(len(data_list[-1])):
      #   material = database["Entry"][j]["Seq"][0]["XS"]["material2"]["X"]
      #   if material == [[0.7],[0.2],[0.0],[0.1]]: materials.append("bounce")
      #   elif material == [[0.1],[0.2],[0.0],[0.1]]: materials.append("nobounce")
      #   elif material == [[0.1],[0.01],[1.5],[0.1]]: materials.append("natto")
      #   elif material == [[0.1],[0.01],[0.25],[0.2]]: materials.append("ketchup")
      # materials = np.array(materials)
      # materials_list.append(materials)
      i += 1
    except:
      break
  save_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  # f = open(save_path+args[0]+"_materials_list.txt", "wb")
  # pickle.dump(materials_list, f)
  f = open(save_path+args[0]+"_materials_list.txt", "rb")
  materials_list = pickle.load(f)

  # database_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" \
  #                 + args[0] + str(11311) + "/database.yaml"
  # with open(database_path) as f:
  #   database = yaml.safe_load(f)
  # # materials = [database["Entry"][i]["Seq"][0]["XS"]["material2"]["X"] for i in range(len(data_list[-1]))]
  # materials = []
  # for i in range(len(data_list[-1])):
  #   material = database["Entry"][i]["Seq"][0]["XS"]["material2"]["X"]
  #   if material == [[0.7],[0.2],[0.0],[0.1]]: materials.append("bounce")
  #   elif material == [[0.1],[0.2],[0.0],[0.1]]: materials.append("nobounce")
  #   elif material == [[0.1],[0.01],[1.5],[0.1]]: materials.append("natto")
  #   elif material == [[0.1],[0.01],[0.25],[0.2]]: materials.append("ketchup")
  # materials = np.array(materials)

  df_list = []
  df_ma_list = []
  df_bounce_list = []
  df_nobounce_list = []
  df_natto_list = []
  df_ketchup_list = []
  window = 10
  for i, (data, materials) in enumerate(zip(data_list,materials_list)):
    skill = [0]*len(data)
    title_reward = "reward_" + args[0] + str(i)
    title_ma = "reward_ma" + str(window) + "_"  + args[0] + str(i)
    df_tmp = pd.DataFrame({title_reward:data[:,1]})
    df_list.append(df_tmp)
    df_ma_list.append(df_tmp.rolling(window).mean())
    df_bounce_list.append(pd.DataFrame({title_reward:data[:,1]*np.where(materials=="bounce",1,np.nan)}))
    df_nobounce_list.append(pd.DataFrame({title_reward:data[:,1]*np.where(materials=="nobounce",1,np.nan)}))
    df_natto_list.append(pd.DataFrame({title_reward:data[:,1]*np.where(materials=="natto",1,np.nan)}))
    df_ketchup_list.append(pd.DataFrame({title_reward:data[:,1]*np.where(materials=="ketchup",1,np.nan)}))
    if i==0: 
      df = df_list[-1]
      df_ma = df_ma_list[-1]
      df_bounce = df_bounce_list[-1]
      df_nobounce = df_nobounce_list[-1]
      df_natto = df_natto_list[-1]
      df_ketchup = df_ketchup_list[-1]
    else: 
      df = pd.concat([df, df_list[-1]], axis=1)
      df_ma = pd.concat([df_ma, df_ma_list[-1]], axis=1)
      df_bounce = pd.concat([df_bounce, df_bounce_list[-1]], axis=1)
      df_nobounce = pd.concat([df_nobounce, df_nobounce_list[-1]], axis=1)
      df_natto = pd.concat([df_natto, df_natto_list[-1]], axis=1)
      df_ketchup = pd.concat([df_ketchup, df_ketchup_list[-1]], axis=1)
  df["mean"] = df.mean(axis=1)
  df["std"] = df.std(axis=1)
  df_ma["mean"] = df_ma.mean(axis=1)
  df_ma["std"] = df_ma.std(axis=1)
  df_bounce["mean"] = df_bounce.mean(axis=1)
  df_bounce["std"] = df_bounce.std(axis=1)
  df_nobounce["mean"] = df_nobounce.mean(axis=1)
  df_nobounce["std"] = df_nobounce.std(axis=1)
  df_natto["mean"] = df_natto.mean(axis=1)
  df_natto["std"] = df_natto.std(axis=1)
  df_ketchup["mean"] = df_ketchup.mean(axis=1)
  df_ketchup["std"] = df_ketchup.std(axis=1)
  df_mtr_list = [df_bounce,df_nobounce,df_natto,df_ketchup]

  if mode=="mtr_mean":
    fig = plt.figure(figsize=(20,5))
    fig.suptitle(args[0])
    PlotMeanStd(fig.add_subplot(1,4,1), df_bounce, "reward (bounce)")
    PlotMeanStd(fig.add_subplot(1,4,2), df_nobounce, "reward (nobounce)")
    PlotMeanStd(fig.add_subplot(1,4,3), df_natto, "reward (natto)")
    PlotMeanStd(fig.add_subplot(1,4,4), df_ketchup, "reward (ketchup)")
    plt.show()
  elif mode=="mtr_all":
    fig = plt.figure(figsize=(20,5))
    fig.suptitle(args[0])
    Scatter(fig.add_subplot(1,4,1), df_bounce.drop(columns=["mean","std"]).iloc[:,:], "reward (bounce)")
    Scatter(fig.add_subplot(1,4,2), df_nobounce.drop(columns=["mean","std"]).iloc[:,:], "reward (nobounce)")
    Scatter(fig.add_subplot(1,4,3), df_natto.drop(columns=["mean","std"]).iloc[:,:], "reward (natto)")
    Scatter(fig.add_subplot(1,4,4), df_ketchup.drop(columns=["mean","std"]).iloc[:,:], "reward (ketchup)")
    plt.show()
  elif mode=="ratio":
    ratio_list = []
    for df in [df_bounce,df_nobounce,df_natto,df_ketchup]:
      tmp_list = []
      total = ((df==df)).sum().sum()
      for i in range(len(df)):
        tmp_list.append(float(((df.iloc[0:i]==df.iloc[0:i])).sum().sum())/total)
      ratio_list.append(tmp_list)
    df_matio = pd.DataFrame({"bounce": ratio_list[0], 
                              "nobounce": ratio_list[1], 
                              "natto": ratio_list[2], 
                              "ketchup": ratio_list[3]})
    Plot(df_ratio,"ratio",0,1)
  elif mode=="org_mean":
    fig = plt.figure(figsize=(20,5))
    fig.suptitle(args[0])
    ax = fig.add_subplot(1,1,1)
    PlotMeanStd(ax,df,"org")
  elif mode=="ma_mean":
    fig = plt.figure(figsize=(20,5))
    fig.suptitle(args[0])
    ax = fig.add_subplot(1,1,1)
    PlotMeanStd(ax,df_ma,"org")