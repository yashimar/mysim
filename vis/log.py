import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pickle
from core_tool import *

def Help():  
  return '''Visualize dpl logs.
  Usage: mysim.vis_log'''

def Plot(df, ylabel):
  for i in range(len(df.columns)):
    plt.plot(df.iloc[:,i], label=df.columns[i])
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    plt.ylim(-30,0)
    plt.legend()
    plt.grid()
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
  name_log = args[0]
  is_ma = args[1] if len(args)==2 else "normal"
  i = 0;
  data_list = []
  log_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" \
                  + name_log + "/dpl_est.dat"
  data_list.append(np.loadtxt(log_path, comments='!'))

  window = 10
  df_list = []
  df_ma_list = []
  for i, data in enumerate(data_list):
    skill = [0]*len(data)
    title_reward = "reward_" + args[0] + str(i)
    title_ma = "reward_ma" + str(window) + "_"  + args[0] + str(i)
    df_tmp = pd.DataFrame({title_reward:data[:,1]})
    df_list.append(df_tmp)
    df_ma_list.append(df_tmp.rolling(window).mean())
    if i==0: 
      df = df_list[-1]
      df_ma = df_ma_list[-1]
    else: 
      df = pd.concat([df, df_list[-1]], axis=1)
      df_ma = pd.concat([df_ma, df_ma_list[-1]], axis=1)
  # df["mean"] = df.mean(axis=1)
  # df["std"] = df.std(axis=1)
  # df_ma["mean"] = df_ma.mean(axis=1)
  # df_ma["std"] = df_ma.std(axis=1)
  
  fig = plt.figure(figsize=(20,5))
  plt.title(args[0]) 
  if is_ma=="ma": Plot(df_ma.iloc[:,:], "reward_ma" + str(window))
  else: Plot(df.iloc[:,:], "reward")
  # PlotMeanStd(df_ma["mean"], df_ma["std"], "reward_ma" + str(window))