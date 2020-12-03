from core_tool import *
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib
from scipy.spatial import distance
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from collections import defaultdict

def Help():  
  return '''Visualize dpl logs.
  Usage: mysim.vis_log'''

def Run(ct, *args):
  name_log = args[0]
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  sl_path = root_path + name_log + "/sequence_list.yaml"
  tree_path = root_path+name_log+"/best_est_trees/"

  with open(sl_path, "r") as yml:
    sl = yaml.load(yml)

  trues = defaultdict(list)
  for ep in range(len(sl)):
    trues["da_spill2"].append(sl[ep]["reward"][3][2][0]["da_spill2"])
    trues["da_pour"].append(sl[ep]["reward"][3][3][0]["da_pour"])
    trues["p_pour_x"].append(sl[ep]["sequence"][2]["n2a"]["p_pour"][0][0])
    trues["p_pour_z"].append(sl[ep]["sequence"][2]["n2a"]["p_pour"][2][0])
    trues["lp_pour_x"].append(sl[ep]["sequence"][3]["n2b"]["lp_pour"][0][0])
    trues["lp_pour_z"].append(sl[ep]["sequence"][3]["n2b"]["lp_pour"][2][0])

  ests = defaultdict(list)
  for i in range(len(sl)):
    with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
      tree = joblib.load(f)
      ests["da_spill2"].append(tree.Tree[TPair("n4sar",0)].XS["da_spill2"].X.item())
      ests["da_pour"].append(tree.Tree[TPair("n4sar",0)].XS["da_pour"].X.item())
      ests["p_pour_x"].append(tree.Tree[TPair("n2a",0)].XS["p_pour"].X[0].item())
      ests["p_pour_z"].append(tree.Tree[TPair("n2a",0)].XS["p_pour"].X[2].item())
      ests["lp_pour_x"].append(tree.Tree[TPair("n2b",0)].XS["lp_pour"].X[0].item())
      ests["lp_pour_z"].append(tree.Tree[TPair("n2b",0)].XS["lp_pour"].X[2].item())


  if False:
    var_list = ["da_spill2", "da_pour", "p_pour_x", "p_pour_z", "lp_pour_x", "lp_pour_z"]
    # var_list = ["da_spill2"]
    ep_block = 20
    for i, var in enumerate(var_list):
      fig = plt.figure(figsize=(20,5))
      fig.suptitle(str(var)+" est/true")
      for j in range(int(len(sl)/ep_block)):
        est = ests[var][j*ep_block:(j+1)*ep_block]
        true = trues[var][j*ep_block:(j+1)*ep_block]
        corr = round(np.corrcoef(est, true)[0][1], 2)
        mse = round(mean_squared_error(true, est), 2)
        fig.add_subplot(1, int(len(sl)/ep_block), j+1).scatter(x=est, y=true)
        plt.title("episode" + str(j*ep_block) + "~" + str((j+1)*ep_block) + "\n" 
                    + "Correlation coefficient: "+str(corr) + "\n"
                    + "MSE: "+str(mse))
        plt.xlabel("est "+var)
        plt.ylabel("true "+var)
        plt.plot(np.linspace(min(min(est), min(true)), max(max(est), max(true)), 2), np.linspace(min(min(est), min(true)), max(max(est), max(true)), 2), c="orange", linestyle="dashed")
        plt.legend()
      plt.subplots_adjust(left=0.05, right=0.95, top=0.8)
      plt.show()
      # plt.savefig("/home/yashima/Pictures/BottomUp//output_est_true/"+str(var)+""+".png")
      # plt.close()

  if False:
    var_list = ["da_spill2", "da_pour", "p_pour_x", "p_pour_z", "lp_pour_x", "lp_pour_z"]
    # var_list = ["da_spill2"]
    for var in var_list:
      fig = plt.figure(figsize=(20,5))
      fig.suptitle(str(var)+" epsiode/error")
      ax = fig.add_subplot(1, 1, 1)
      est = ests[var]
      true = trues[var]
      diff = [abs(t-e) for t,e in zip(true, est)]
      ax.plot(diff)
      ax.set_xlim(0,len(diff))
      ax.set_xticks(np.arange(0, len(diff)+1, 10))
      ax.set_xticks(np.arange(0, len(diff)+1, 1), minor=True)
      ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
      ax.grid(which='major', alpha=0.9, linestyle='dotted') 
      plt.xlabel("episode")
      plt.ylabel("|true - est|")
      plt.subplots_adjust(left=0.05, right=0.95, top=0.8)
      plt.show()
      # plt.savefig("/home/yashima/Pictures/BottomUp//output_episode_error/"+str(var)+""+".png")
      # plt.close()