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

  envs = defaultdict(list)
  trues = defaultdict(list)
  for ep in range(len(sl)):
    config = sl[ep]["config"]
    reward = sl[ep]["reward"]
    sequence = sl[ep]["sequence"]

    envs["smsz"].append(config["size_srcmouth"][0][0])
    if config["material2"][0][0] == 0.7: envs["mtr"].append("bounce")
    elif config["material2"][2][0] == 0.25: envs["mtr"].append("ketchup")
    elif config["material2"][0][0] == 1.5: envs["mtr"].append("natto")
    else: envs["mtr"].append("nobounce")
    trues["da_spill2"].append(reward[1][2][0]["da_spill2"])
    trues["da_pour"].append(reward[1][3][0]["da_pour"])
    trues["lp_pour_x"].append(sequence[3]["n2b"]["lp_pour"][0][0])
    trues["lp_pour_z"].append(sequence[3]["n2b"]["lp_pour"][2][0])

  ests = defaultdict(list)
  ests_cov = defaultdict(list)
  for i in range(len(sl)):
    with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
      tree = joblib.load(f)
      skill = tree.Tree[TPair("n2c",0)].XS["skill"].X.item()
      if skill==0:
        ests["da_spill2"].append(tree.Tree[TPair("n4tir",0)].XS["da_spill2"].X.item())
        ests["da_pour"].append(tree.Tree[TPair("n4tir",0)].XS["da_pour"].X.item())
      elif skill==1:
        ests["da_spill2"].append(tree.Tree[TPair("n4sar",0)].XS["da_spill2"].X.item())
        ests_cov["da_spill2"].append(tree.Tree[TPair("n4sar",0)].XS["da_spill2"].Cov.item())
        ests["da_pour"].append(tree.Tree[TPair("n4sar",0)].XS["da_pour"].X.item())
        ests_cov["da_pour"].append(tree.Tree[TPair("n4sar",0)].XS["da_pour"].Cov.item())
      # ests["p_pour_x"].append(tree.Tree[TPair("n2a",0)].XS["p_pour"].X[0].item())
      # ests["p_pour_z"].append(tree.Tree[TPair("n2a",0)].XS["p_pour"].X[2].item())
      ests["lp_pour_x"].append(tree.Tree[TPair("n2b",0)].XS["lp_pour"].X[0].item())
      ests["lp_pour_z"].append(tree.Tree[TPair("n2b",0)].XS["lp_pour"].X[2].item())

  # var_list = [ "lp_pour_x", "lp_pour_z"]
  var_list = ["da_spill2"]

  if True:
    ep_block = 40
    for i, var in enumerate(var_list):
      # smsz = 0.08
      fig = plt.figure(figsize=(20,4))
      fig.suptitle(str(var)+" est/true", fontsize=15)
      # fig.suptitle(str(var)+" est/true (smsz: "+str(smsz-0.01)+"~"+str(smsz)+")", fontsize=15)
      n_row = 1
      n_col = int(len(sl)/ep_block)
      for j in range(int(len(sl)/ep_block)):
        est = ests[var][j*ep_block:(j+1)*ep_block]
        true = trues[var][j*ep_block:(j+1)*ep_block]
        # smsz_ids = [i for i, x in enumerate(envs["smsz"]) if smsz-0.01<x<smsz and j*ep_block<=i<(j+1)*ep_block]
        # est = np.array(ests[var])[smsz_ids]
        # true = np.array(trues[var])[smsz_ids]
        corr = round(np.corrcoef(est, true)[0][1], 2)
        rmse = round(np.sqrt(mean_squared_error(true, est)), 2)
        diff = abs(np.array(true)-np.array(est))
        max_diff = round(max(diff), 2)
        q_diff = map(lambda x: round(x,2), np.percentile(diff, q=[25,50,75]))
        fig.add_subplot(n_row, n_col, j+1).scatter(x=est, y=true)
        plt.title("episode" + str(j*ep_block) + "~" + str((j+1)*ep_block) + "\n" 
                    + "Correlation coefficient: "+str(corr) + "\n"
                    + "RMSE: "+str(rmse) + "\n"
                    # + "max |true-est|: "+str(max_diff) + "\n"
                    # + "quantile |true-est|: "+str(", ".join(map(str, q_diff))) 
                  )
        plt.xlabel("est "+var)
        plt.ylabel("true "+var)
        plt.plot(np.linspace(min(min(est), min(true)), max(max(est), max(true)), 2), np.linspace(min(min(est), min(true)), max(max(est), max(true)), 2), c="orange", linestyle="dashed")
        # plt.plot(np.linspace(0.2, 0.5, 2), np.linspace(0.2, 0.5, 2), c="orange", linestyle="dashed")
        # plt.xlim(0.2,0.5)
        # plt.ylim(0.2,0.5)
        plt.legend()
      plt.subplots_adjust(left=0.05, right=0.95, top=0.70)
      plt.show()
      # plt.savefig("/home/yashima/Pictures/BottomUp/learn5_MaterialOnlyShake/after_add_5FixedPPourTrgSample/output_est_true/"+str(var)+""+".png")
      # plt.close()

  if False:
    max_dict = {
      "da_spill2": 0.5, 
      "da_pour": 0.1
    }
    for var in var_list:
      fig = plt.figure(figsize=(20,4))
      fig.suptitle(str(var)+" epsiode/error")
      ax = fig.add_subplot(1, 1, 1)
      est = ests[var]
      true = trues[var]
      diff = [abs(t-e) for t,e in zip(true, est)]
      ax.plot(diff, label="diff")
      # c_dict = {"bounce":"purple","nobounce":"green","natto":"orange","ketchup":"red"}
      # for mtr in list(set(envs["mtr"])):
      #   mtr_ids = [i for i, x in enumerate(envs["mtr"]) if x==mtr]
      #   ax.scatter(mtr_ids, np.array(diff)[mtr_ids], label=mtr, c=c_dict[mtr])
      plt.legend()
      ax.set_xlim(0,len(diff))
      ax.set_xlim(0,0.5)
      ax.set_ylim(0,max_dict[var])
      ax.set_xticks(np.arange(0, len(diff)+1, 10))
      ax.set_xticks(np.arange(0, len(diff)+1, 1), minor=True)
      ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
      ax.grid(which='major', alpha=0.9, linestyle='dotted') 
      plt.xlabel("episode")
      plt.ylabel("|true - est|")
      plt.subplots_adjust(left=0.05, right=0.95, top=0.9)
      plt.show()
      # plt.savefig("/home/yashima/Pictures/BottomUp/learn5_MaterialOnlyShake/after_add_5FixedPPourTrgSample/output_episode_error/"+str(var)+""+".png")
      # plt.close()

  if False:
    max_dict = {
      "da_spill2": 0.5, 
      "da_pour": 0.1
    }
    for var in var_list:
      fig = plt.figure(figsize=(20,4))
      fig.suptitle(str(var)+" epsiode/error")
      ax = fig.add_subplot(1, 1, 1)
      est = ests[var]
      est_cov = ests_cov[var]
      true = trues[var]
      # ax.plot(np.array(true)/100, label="true")
      ax.plot(est_cov, label="sdv estimation")
      # c_dict = {"bounce":"purple","nobounce":"green","natto":"orange","ketchup":"red"}
      # for mtr in list(set(envs["mtr"])):
      #   mtr_ids = [i for i, x in enumerate(envs["mtr"]) if x==mtr]
      #   ax.scatter(mtr_ids, np.array(diff)[mtr_ids], label=mtr, c=c_dict[mtr])
      plt.legend()
      ax.set_xlim(0,len(est))
      ax.set_ylim(0,0.1)
      ax.set_xticks(np.arange(0, len(est)+1, 10))
      ax.set_xticks(np.arange(0, len(est)+1, 1), minor=True)
      ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
      ax.grid(which='major', alpha=0.9, linestyle='dotted') 
      plt.xlabel("episode")
      # plt.ylabel("|true - est|")
      plt.subplots_adjust(left=0.05, right=0.95, top=0.9)
      plt.show()
      # plt.savefig("/home/yashima/Pictures/BottomUp/learn5_MaterialOnlyShake/after_add_5FixedPPourTrgSample/output_episode_error/"+str(var)+""+".png")
      # plt.close()

  if False:
    max_dict = {
      "da_spill2": 0.7, 
      "da_pour": 0.2
    }
    WindowSize = 20
    for var in var_list:
      est = ests[var]
      true = trues[var]
      diff = [abs(t-e) for t,e in zip(true, est)]
      diff_max = [None]*WindowSize + [max(diff[i-WindowSize:i]) for i in range(WindowSize,len(est))]
      diff_q1 = [None]*WindowSize + [np.percentile(diff[i-WindowSize:i], [25]) for i in range(WindowSize,len(est))]
      diff_q2 = [None]*WindowSize + [np.percentile(diff[i-WindowSize:i], [50]) for i in range(WindowSize,len(est))]
      diff_q3 = [None]*WindowSize + [np.percentile(diff[i-WindowSize:i], [75]) for i in range(WindowSize,len(est))]

      fig = plt.figure(figsize=(20,4))
      fig.suptitle(str(var)+" epsiode/error max and quantile (window: "+str(WindowSize)+")", fontsize=15)
      ax = fig.add_subplot(1, 1, 1)
      ax.plot(diff_max, label="max")
      ax.plot(diff_q1, label="25%")
      ax.plot(diff_q2, label="50%")
      ax.plot(diff_q3, label="75%")
      plt.legend()
      ax.set_xlim(0,len(diff))
      ax.set_xlim(0,0.5)
      ax.set_ylim(0,max_dict[var])
      ax.set_xticks(np.arange(0, len(diff)+1, 10))
      ax.set_xticks(np.arange(0, len(diff)+1, 1), minor=True)
      ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
      ax.grid(which='major', alpha=0.9, linestyle='dotted') 
      plt.xlabel("episode")
      plt.ylabel("filltered |true - est|")
      plt.subplots_adjust(left=0.05, right=0.95, top=0.8)
      plt.show()
      # plt.savefig("/home/yashima/Pictures/BottomUp/learn5_MaterialOnlyShake/after_add_5FixedPPourTrgSample/output_episode_error/"+str(var)+""+".png")
      # plt.close()