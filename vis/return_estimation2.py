import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from collections import defaultdict
import yaml
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

  data_list = []
  with open(log_path, "r") as log_data:
    for ep, line in enumerate(log_data):
      line = line.split("\n")[0].split(" ")
      line = map(lambda x: float(x), line)
      data_list.append(line)
  returns = defaultdict(list)
  for i, data in enumerate(data_list):
    returns["eval"].append(data[1])
    returns["est_n0"].append(data[3])

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

  eval = returns["eval"]
  est = returns["est_n0"]

  if True:
    eval = returns["eval"]
    est = returns["est_n0"]

    fig = plt.figure(figsize=(20,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("eval/est(n0) return", fontsize=15)

    plt.ylabel("- return")
    # ax.set_ylim(-1,0)
    eval = -1*np.array(eval)
    est = -1*np.array(est)
    diff = abs(eval - est)
    ax.set_yscale('log')
    # plt.ylabel("log(-return)")
    ax.set_ylim(0.0001,100)

    # ax.axhline(y=0.25, xmin=0, xmax=len(eval), c="purple",linewidth=1,linestyle="dashed", label="return = -0.25")
    # ax.axhline(y=0.1, xmin=0, xmax=len(eval), c="red",linewidth=1,linestyle="dashed", label="return = -0.1")

    ax.plot(eval, label="eval")
    ax.plot(est, label="est(n0)", c="pink")
    # c_dict = {"bounce":"purple","nobounce":"green","natto":"orange","ketchup":"red"}
    # for mtr in list(set(envs["mtr"])):
    #   mtr_ids = [i for i, x in enumerate(envs["mtr"]) if x==mtr]
    #   ax.scatter(mtr_ids, np.array(eval)[mtr_ids], label=mtr, c=c_dict[mtr])
    c_dict = {"std_pour":"green","shake_A":"red"}
    for skill in list(set(skills)):
      skill_ids = [i for i, x in enumerate(skills) if x==skill]
      ax.scatter(skill_ids, np.array(eval)[skill_ids], label=skill, c=c_dict[skill])
    # c_dict = {"bounce":"purple","nobounce":"green","natto":"orange","ketchup":"red"}
    # marker_dict = {"std_pour":"o","shake_A":"*"}
    # for mtr in list(set(envs["mtr"])):
    #   mtr_ids = [i for i, x in enumerate(envs["mtr"]) if x==mtr]
    #   for skill in list(set(skills)):
    #     skill_ids = [i for i, x in enumerate(skills) if x==skill]
    #     ids = list(set(mtr_ids) & set(skill_ids))
    #     ax.scatter(ids, np.array(eval)[ids], label=mtr, c=c_dict[mtr], marker=marker_dict[skill])
    ax.set_xlim(0,len(eval))
    ax.set_xticks(np.arange(0, len(eval)+1, 10))
    ax.set_xticks(np.arange(0, len(eval)+1, 1), minor=True)
    ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
    ax.grid(which='major', alpha=0.9, linestyle='dotted') 
    plt.xlabel("episode")
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()


  if False:
    eval = returns["eval"]
    est = returns["est_n0"]

    fig = plt.figure(figsize=(20,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("return estimation error", fontsize=15)

    diff = abs(np.array(eval) - np.array(est))
    # ax.set_yscale('log')
    # plt.ylabel("log(|eval - est|)")
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
    ax.set_xlim(0,len(eval))
    ax.set_xticks(np.arange(0, len(diff)+1, 10))
    ax.set_xticks(np.arange(0, len(diff)+1, 1), minor=True)
    ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
    ax.grid(which='major', alpha=0.9, linestyle='dotted') 
    plt.xlabel("episode")
    plt.ylabel("|eval - est|")
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()


  if False:    
    eval = returns["eval"]
    est = returns["est_n0"]

    eval = -1*np.array(eval)
    est = -1*np.array(est)
  
    # fig = plt.figure(figsize=(20,4))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_title("eval/est return", fontsize=15)

    # plt.ylabel("return")
    # ax.set_ylim(-0.3,0)
    # eval = -1*np.array(eval)
    # est = -1*np.array(est)
    # diff = abs(eval - est)
    # ax.set_yscale('log')
    # plt.ylabel("log(-return)")
    # ax.set_ylim(0.01,1)

    # ax.plot(eval, label="eval")
    # ax.plot(est, label="est", c="orange")
    # c_dict = {0.04:"purple", 0.05:"green", 0.06:"blue", 0.07:"orange", 0.08:"red"}
    c_dict = {"std_pour":"green","shake_A":"red"}
    for smsz in [0.04,0.05,0.06,0.07,0.08]:
      fig = plt.figure(figsize=(20,4))
      ax = fig.add_subplot(1, 1, 1)
      ax.set_title("eval return (smsz: "+str(smsz-0.01)+"~"+str(smsz)+")", fontsize=15)

      # plt.ylabel("return")
      # ax.set_ylim(-10.0,0)
      plt.ylabel("- return")
      ax.set_yscale('log')
      ax.set_ylim(0.0001,100)

      smsz_ids = [i for i, x in enumerate(envs["smsz"]) if smsz-0.01<x<smsz]
      for skill in list(set(skills)):
        skill_ids = [i for i, x in enumerate(skills) if x==skill]
        ids = list(set(smsz_ids) & set(skill_ids))
        ax.scatter(ids, np.array(eval)[ids], label=skill, c=c_dict[skill])

      # ax.scatter(smsz_ids, np.array(eval)[smsz_ids], label=smsz, c=c_dict[smsz])
      ax.set_xlim(0,len(eval))
      ax.set_xticks(np.arange(0, len(eval)+1, 10))
      ax.set_xticks(np.arange(0, len(eval)+1, 1), minor=True)
      ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
      ax.grid(which='major', alpha=0.9, linestyle='dotted') 
      plt.xlabel("episode")
      # plt.legend(loc='lower right')
      plt.legend()
      plt.subplots_adjust(left=0.05, right=0.95)
      plt.show()