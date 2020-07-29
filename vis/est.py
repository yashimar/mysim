import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  mtr = args[0]
  base_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/try_dpl/"

  fig = plt.figure(figsize=(10,5))
  for i,sms in enumerate(["002","0055","009"]):
    ax = fig.add_subplot(1,3,i+1)
    data = []
    for skill in ["std_pour","shake_A","choose"]:  
      name = mtr+"_"+sms
      data_path = base_path + skill + "/" + name + "/dpl_est.dat"
      data.append(np.loadtxt(data_path, comments='!')[:,1])
    ax.boxplot(data)
    ax.set_title(mtr+"_"+sms)
    ax.set_xticklabels(["std_pour","shake_A","choose"])
  plt.show()

  
