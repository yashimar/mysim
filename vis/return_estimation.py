import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from collections import defaultdict
from core_tool import *

def Help():
  pass

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

def Run(ct, *args):
  name_log = args[0]
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  log_path = root_path + name_log + "/dpl_est.dat"

  data_list = []
  with open(log_path, "r") as log_data:
    for ep, line in enumerate(log_data):
      line = line.split("\n")[0].split(" ")
      line = map(lambda x: float(x), line)
      data_list.append(line)
  returns = defaultdict(list)
  for i, data in enumerate(data_list):
    returns["true"].append(data[1])
    returns["est_n0"].append(data[2])
  

  true = returns["true"]
  est = returns["est_n0"]

  fig = plt.figure(figsize=(20,5))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_title(args[0]) 
  ax.plot(true, label="true")
  ax.plot(est, label="est", c="orange")
  ax.set_xlim(0,len(true))
  ax.set_ylim(-30,0)
  ax.set_xticks(np.arange(0, len(true)+1, 10))
  ax.set_xticks(np.arange(0, len(true)+1, 1), minor=True)
  ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
  ax.grid(which='major', alpha=0.9, linestyle='dotted') 
  plt.xlabel("episode")
  plt.ylabel("return")
  plt.legend()
  plt.subplots_adjust(left=0.05, right=0.95)
  plt.show()