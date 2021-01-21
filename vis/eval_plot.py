import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from collections import defaultdict
import yaml
import joblib
from core_tool import *

def Help():
  pass

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
    returns["eval"].append(data[1])
    # returns["est_n0"].append(data[3])

  with open(sl_path, "r") as yml:
    sl = yaml.load(yml)

  envs = defaultdict(list)
  evals = defaultdict(list)
  skills = []
  for ep in range(len(sl)):
    config = sl[ep]["config"]
    seq = sl[ep]["sequence"]
    reward = sl[ep]["reward"]

    envs["smsz"].append(config["size_srcmouth"][0][0])
    if config["material2"][0][0] == 0.7: envs["mtr"].append("bounce")
    elif config["material2"][2][0] == 0.25: envs["mtr"].append("ketchup")
    elif config["material2"][2][0] == 1.5: envs["mtr"].append("natto")
    else: envs["mtr"].append("nobounce")
    if "sa" in seq[4].keys()[0]: skills.append("shake_A")
    else:              skills.append("std_pour")
    evals["a_pour"].append(reward[-1][4][0]["a_pour"])
    evals["da_pour"].append(reward[-1][3][0]["da_pour"])
    evals["a_spill2"].append(reward[-1][5][0]["a_spill2"])
    evals["da_spill2"].append(reward[-1][2][0]["da_spill2"])
    evals["eval"].append(-100*max(0.3-evals["a_pour"][-1],0)**2 -1.0*max(evals["a_pour"][-1]-0.3,0)**2 -1.0*max(evals["a_spill2"][-1],0)**2)

  ests = defaultdict(list)
  ests_sdv = defaultdict(list)
  for i in range(len(sl)):
    with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
      tree = joblib.load(f)
      skill = tree.Tree[TPair("n2c",0)].XS["skill"].X.item()
      # if True:
      if skill==0:
        ests["da_spill2"].append(tree.Tree[TPair("n4tir",0)].XS["da_spill2"].X.item())
        ests_sdv["da_spill2"].append(np.sqrt(tree.Tree[TPair("n4tir",0)].XS["da_spill2"].Cov.item()))
        ests["da_pour"].append(tree.Tree[TPair("n4tir",0)].XS["da_pour"].X.item())
        ests_sdv["da_pour"].append(np.sqrt(tree.Tree[TPair("n4tir",0)].XS["da_pour"].Cov.item()))
      elif skill==1:
        ests["da_spill2"].append(tree.Tree[TPair("n4sar",0)].XS["da_spill2"].X.item())
        ests_sdv["da_spill2"].append(np.sqrt(tree.Tree[TPair("n4sar",0)].XS["da_spill2"].Cov.item()))
        ests["da_pour"].append(tree.Tree[TPair("n4sar",0)].XS["da_pour"].X.item())
        ests_sdv["da_pour"].append(np.sqrt(tree.Tree[TPair("n4sar",0)].XS["da_pour"].Cov.item()))
      # ests["p_pour_x"].append(tree.Tree[TPair("n2a",0)].XS["p_pour"].X[0].item())
      # ests["p_pour_z"].append(tree.Tree[TPair("n2a",0)].XS["p_pour"].X[2].item())
      ests["lp_pour_x"].append(tree.Tree[TPair("n2b",0)].XS["lp_pour"].X[0].item())
      ests["lp_pour_z"].append(tree.Tree[TPair("n2b",0)].XS["lp_pour"].X[2].item())

  # print(ests_sdv["da_pour"])
  # print(len(ests_sdv["da_pour"]))
  # print(ests["da_pour"])
  # print(len(ests["da_pour"]))

  lim_dict = {"da_pour":[0,0.6], "da_spill2":[0,1.5]}
  for key in lim_dict.keys():
    fig = plt.figure(figsize=(20,3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("observed "+key+", skill=sfd_pour")
    # ax.plot(np.linspace(0.03,0.08,len(evals[key])), evals[key], label=key)
    ax.plot(np.linspace(0,len(evals[key])-1,len(evals[key])), evals[key], label=key)
    ax.errorbar(np.linspace(0,len(ests[key])+1, len(ests[key])), ests[key], label="est", c="pink", yerr=ests_sdv[key], fmt='*', markersize=4, zorder=-1)
    if key=="a_pour": ax.axhline(y=0.3, label="target amount", c="red")
    elif key=="a_spill2": ax.axhline(y=1.0, label="spilled_stop", c="red")
    # ax.set_xlim(0.03,0.08)
    ax.set_ylim(lim_dict[key][0], lim_dict[key][1])
    ax.set_xlabel("smsz")
    ax.set_ylabel(key)
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, hspace=0.6)
    plt.show()

  # fig = plt.figure(figsize=(20,4))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.set_title("observed a_pour around smsz=0.07, p_pour_trg=(0.538,0.535), skill=sfd_pour")
  # ax.plot(np.linspace(0.065,0.075,len(a_pour)), a_pour, label="a_pour")
  # ax.axhline(y=0.3, label="target amount", c="red")
  # ax.set_xlim(0.065,0.075)
  # plt.legend()
  # plt.show()

  # fig = plt.figure(figsize=(20,4))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.set_title("observed a_spill2 around smsz=0.07, p_pour_trg=(0.538,0.535), skill=sfd_pour")
  # ax.plot(np.linspace(0.065,0.075,len(a_pour)), a_spill2, label="a_pour")
  # # ax.axhline(y=0.3, label="target amount", c="red")
  # ax.set_xlim(0.065,0.075)
  # plt.legend()
  # plt.show()

  # fig = plt.figure(figsize=(20,4))
  # ax = fig.add_subplot(1, 1, 1)
  # ax.set_title("observed a_spill2 around smsz=0.07, p_pour_trg=(0.538,0.535), skill=sfd_pour")
  # ax.plot(np.linspace(0.065,0.075,len(a_pour)), a_spill2, label="a_pour")
  # # ax.axhline(y=0.3, label="target amount", c="red")
  # ax.set_xlim(0.065,0.075)
  # plt.legend()
  # plt.show()