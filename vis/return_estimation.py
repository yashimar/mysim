from pickle import FALSE
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from collections import defaultdict
import yaml
import joblib
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
  sl_path = root_path + name_log + "/sequence_list.yaml"
  tree_path = root_path+name_log+"/best_est_trees/"

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

  r_est_mean = defaultdict(list)
  r_est_sdv = defaultdict(list)
  for i in range(len(data_list)):
    with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
      tree = joblib.load(f)
      skill = 0
      # print(tree.Tree.keys())
      for key in tree.Tree.keys():
        if key.A == "n0":
          skill = tree.Tree[key].XS["skill"].X
          # pass
        elif key.A == "n4tir":
          r_ti_x = tree.Tree[key].XS[".r"].X.item()
          r_ti_sdv = np.sqrt(tree.Tree[key].XS[".r"].Cov.item())
        elif key.A == "n4sar":
          r_sa_x = tree.Tree[key].XS[".r"].X.item()
          r_sa_sdv = np.sqrt(tree.Tree[key].XS[".r"].Cov.item())
      if skill==0:
        r_est_mean["selected"].append(r_ti_x)
        r_est_mean["shake_A_selected"].append(None)
        r_est_mean["std_pour_selected"].append(r_ti_x)
        r_est_sdv["selected"].append(r_ti_sdv)
        r_est_sdv["shake_A_selected"].append(None)
        r_est_sdv["std_pour_selected"].append(r_ti_sdv)
      elif skill==1:  
        r_est_mean["selected"].append(r_sa_x)
        r_est_mean["shake_A_selected"].append(r_sa_x)
        r_est_mean["std_pour_selected"].append(None)
        r_est_sdv["selected"].append(r_sa_sdv)
        r_est_sdv["shake_A_selected"].append(r_sa_sdv)
        r_est_sdv["std_pour_selected"].append(None)
      r_est_mean["std_pour"].append(r_ti_x)
      r_est_mean["shake_A"].append(r_sa_x)
      r_est_sdv["std_pour"].append(r_ti_sdv)
      r_est_sdv["shake_A"].append(r_sa_sdv)

  with open(sl_path, "r") as yml:
    sl = yaml.load(yml)

  envs = defaultdict(list)
  skills = []
  for ep in range(len(sl)):
    config = sl[ep]["config"]
    seq = sl[ep]["sequence"]

    envs["smsz"].append(config["size_srcmouth"][0][0])
    if config["material2"][0][0] == 0.7: envs["mtr"].append("bounce")
    elif config["material2"][2][0] == 0.25: envs["mtr"].append("ketchup")
    elif config["material2"][2][0] == 1.5: envs["mtr"].append("natto")
    else: envs["mtr"].append("nobounce")
    if "sa" in seq[4].keys()[0]: skills.append("shake_A")
    else:              skills.append("std_pour")

  true = returns["true"]
  vis_skill = "selected"
  # vis_skill = "std_pour_selected"
  # vis_skill = "shake_A_selected"
  # vis_skill = "std_pour"
  # vis_skill = "shake_A"
  est = r_est_mean[vis_skill]
  sdvs = r_est_sdv[vis_skill]


  if False:
    true = returns["true"]
    est = returns["est_n0"]

    fig = plt.figure(figsize=(20,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("true/est return", fontsize=15)

    plt.ylabel("- return")
    true = -1*np.array(true)
    est = -1*np.array(est)
    ax.set_yscale('log')
    ax.set_ylim(1e-4,1e2)
    # plt.ylabel("return")
    # ax.set_ylim(-5,0)
    ax.plot(true, label="true", zorder=0)
    ax.plot(est, label="estimation")

    ax.set_xlim(0,len(true))
    ax.set_xticks(np.arange(0, len(true)+1, 10))
    ax.set_xticks(np.arange(0, len(true)+1, 1), minor=True)
    ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
    ax.grid(which='major', alpha=0.9, linestyle='dotted') 
    plt.xlabel("episode")
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()

  if True:
    true = returns["true"]
    est = returns["est_n0"]

    fig = plt.figure(figsize=(20,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("true/est return", fontsize=15)

    plt.ylabel("- return")
    true = -1*np.array(true)
    est = -1*np.array(est)
    ax.set_yscale('log')
    ax.set_ylim(1e-4,1e2)
    # plt.ylabel("return")
    # ax.set_ylim(-5,0)

    # ax.axhline(y=0.25, xmin=0, xmax=len(true), c="purple",linewidth=1,linestyle="dashed", label="return = -0.25")
    # ax.axhline(y=0.1, xmin=0, xmax=len(true), c="red",linewidth=1,linestyle="dashed", label="return = -0.1")

    ax.plot(true, label="true", zorder=0)
    ax.plot(est, label="estimation")
    # ax.errorbar(np.linspace(0,len(est)+1, len(est)), est, label="est", c="pink", yerr=sdvs, fmt='*', markersize=4, zorder=-1)
    # c_dict = {"bounce":"purple","nobounce":"green","natto":"orange","ketchup":"red"}
    # for mtr in list(set(envs["mtr"])):
    #   mtr_ids = [i for i, x in enumerate(envs["mtr"]) if x==mtr]
    #   ax.scatter(mtr_ids, np.array(true)[mtr_ids], label=mtr, c=c_dict[mtr])
    c_dict = {"std_pour":"green","shake_A":"red"}
    for skill in list(set(skills)):
      skill_ids = [i for i, x in enumerate(skills) if x==skill]
      ax.scatter(skill_ids, np.array(true)[skill_ids], label=skill, c=c_dict[skill])
    # c_dict = {"bounce":"purple","nobounce":"green","natto":"orange","ketchup":"red"}
    # marker_dict = {"std_pour":"o","shake_A":"*"}
    # for mtr in list(set(envs["mtr"])):
    #   mtr_ids = [i for i, x in enumerate(envs["mtr"]) if x==mtr]
    #   for skill in list(set(skills)):
    #     skill_ids = [i for i, x in enumerate(skills) if x==skill]
    #     ids = list(set(mtr_ids) & set(skill_ids))
    #     ax.scatter(ids, np.array(true)[ids], label=mtr, c=c_dict[mtr], marker=marker_dict[skill])
    ax.set_xlim(0,len(true))
    ax.set_xticks(np.arange(0, len(true)+1, 10))
    ax.set_xticks(np.arange(0, len(true)+1, 1), minor=True)
    ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
    ax.grid(which='major', alpha=0.9, linestyle='dotted') 
    plt.xlabel("episode")
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()


  if False:
    true = returns["true"]
    est = returns["est_n0"]

    fig = plt.figure(figsize=(20,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("return estimation error", fontsize=15)

    diff = abs(np.array(true) - np.array(est))
    # ax.set_yscale('log')
    # plt.ylabel("log(|true - est|)")
    # ax.set_ylim(0.0,1.0)
    ax.set_yscale('log')

    ax.plot(diff, label="diff")
    # c_dict = {"bounce":"purple","nobounce":"green","natto":"orange","ketchup":"red"}
    # for mtr in list(set(envs["mtr"])):
    #   mtr_ids = [i for i, x in enumerate(envs["mtr"]) if x==mtr]
    #   ax.scatter(mtr_ids, np.array(diff)[mtr_ids], label=mtr, c=c_dict[mtr])
    c_dict = {"std_pour":"green","shake_A":"red"}
    for skill in list(set(skills)):
      skill_ids = [i for i, x in enumerate(skills) if x==skill]
      ax.scatter(skill_ids, np.array(diff)[skill_ids], label=skill, c=c_dict[skill])
    ax.set_xlim(0,len(true))
    ax.set_xticks(np.arange(0, len(diff)+1, 10))
    ax.set_xticks(np.arange(0, len(diff)+1, 1), minor=True)
    ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
    ax.grid(which='major', alpha=0.9, linestyle='dotted') 
    plt.xlabel("episode")
    plt.ylabel("|true - est|")
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()


  if False:    
    true = returns["true"]
    est = returns["est_n0"]

    true = -1*np.array(true)
    est = -1*np.array(est)
  
    # fig = plt.figure(figsize=(20,4))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_title("true/est return", fontsize=15)

    # plt.ylabel("return")
    # ax.set_ylim(-0.3,0)
    # true = -1*np.array(true)
    # est = -1*np.array(est)
    # diff = abs(true - est)
    # ax.set_yscale('log')
    # plt.ylabel("log(-return)")
    # ax.set_ylim(0.01,1)

    # ax.plot(true, label="true")
    # ax.plot(est, label="est", c="orange")
    # c_dict = {0.04:"purple", 0.05:"green", 0.06:"blue", 0.07:"orange", 0.08:"red"}
    c_dict = {"std_pour":"green","shake_A":"red"}
    for smsz in [0.04,0.05,0.06,0.07,0.08]:
      fig = plt.figure(figsize=(20,4))
      ax = fig.add_subplot(1, 1, 1)
      ax.set_title("true return (smsz: "+str(smsz-0.01)+"~"+str(smsz)+")", fontsize=15)

      # plt.ylabel("return")
      # ax.set_ylim(-10.0,0)
      plt.ylabel("- return")
      ax.set_yscale('log')
      ax.set_ylim(0.0001,100)

      smsz_ids = [i for i, x in enumerate(envs["smsz"]) if smsz-0.01<x<smsz]
      for skill in list(set(skills)):
        skill_ids = [i for i, x in enumerate(skills) if x==skill]
        ids = list(set(smsz_ids) & set(skill_ids))
        ax.scatter(ids, np.array(true)[ids], label=skill, c=c_dict[skill])

      # ax.scatter(smsz_ids, np.array(true)[smsz_ids], label=smsz, c=c_dict[smsz])
      ax.set_xlim(0,len(true))
      ax.set_xticks(np.arange(0, len(true)+1, 10))
      ax.set_xticks(np.arange(0, len(true)+1, 1), minor=True)
      ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
      ax.grid(which='major', alpha=0.9, linestyle='dotted') 
      plt.xlabel("episode")
      # plt.legend(loc='lower right')
      plt.legend()
      plt.subplots_adjust(left=0.05, right=0.95)
      plt.show()