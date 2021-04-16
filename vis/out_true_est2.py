from pickle import FALSE
from core_tool import *
from matplotlib.pyplot import legend
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
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
  vis_state = args[1]
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  sl_path = root_path + name_log + "/sequence_list.yaml"
  tree_path = root_path+name_log+"/best_est_trees/"
  color_list = args[2] if len(args)==3 else None

  with open(sl_path, "r") as yml:
    sl = yaml.load(yml)

  envs = defaultdict(list)
  trues = defaultdict(list)
  skills = []
  for ep in range(len(sl)):
    config = sl[ep]["config"]
    reward = sl[ep]["reward"]
    sequence = sl[ep]["sequence"]

    envs["smsz"].append(config["size_srcmouth"][0][0])
    if config["material2"][0][0] == 0.7: envs["mtr"].append("bounce")
    elif config["material2"][2][0] == 0.25: envs["mtr"].append("ketchup")
    elif config["material2"][0][0] == 1.5: envs["mtr"].append("natto")
    else: envs["mtr"].append("nobounce")
    if "sa" in sequence[4].keys()[0]: skills.append("shake_A")
    else:                        skills.append("std_pour")
    trues["da_spill2"].append(reward[1][2][0]["da_spill2"])
    trues["da_pour"].append(reward[1][3][0]["da_pour"])
    trues[".r"].append(reward[0]["total"])
    trues["p_pour_trg"].append(sequence[2]["n2a"]["p_pour_trg"])
    trues["lp_pour_x"].append(sequence[3]["n2b"]["lp_pour"][0][0])
    trues["lp_pour_z"].append(sequence[3]["n2b"]["lp_pour"][2][0])
    # trues["da_total"].append(sequence[5][sequence[5].keys()[0]]["da_total"][0][0])

  ests = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  for i in range(len(sl)):
    with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
      tree = joblib.load(f)
      for n in tree.Tree.keys():
        if n.A=="n4tir": n_tir = n
        elif n.A=="n4sar": n_sar = n
      # tir_xs = tree.Tree[n_tir].XS
      sar_xs = tree.Tree[n_sar].XS
      # selected_xs = tir_xs if tir_xs[".r"].X.item()>sar_xs[".r"].X.item() else sar_xs
      for (est_dict, r_xs) in zip([ests["sa"]], [sar_xs]):
      # for (est_dict, r_xs) in zip([ests["ti"]], [tir_xs]):
      # for (est_dict, r_xs) in zip([ests["ti"], ests["sa"], ests["selected"]], [tir_xs, sar_xs, selected_xs]):
        for s in ["da_spill2", "da_pour", ".r"]:
          est_dict[s]["mean"].append(r_xs[s].X.item())
          est_dict[s]["sdv"].append(np.sqrt(r_xs[s].Cov.item()))
        # est_dict["skill"]["mean"].append(r_xs["skill"].X.item())
        # est_dict["lp_pour_x"]["mean"].append(r_xs["lp_pour"].X[0].item())
        # est_dict["lp_pour_x"]["sdv"].append(np.sqrt(r_xs["lp_pour"].Cov[0,0].item()))
        # est_dict["lp_pour_z"]["mean"].append(r_xs["lp_pour"].X[2].item())
        # est_dict["lp_pour_z"]["sdv"].append(np.sqrt(r_xs["lp_pour"].Cov[2,2].item()))
      

  if vis_state=="da_pour":
    ylim = [-1,100] 
  elif vis_state=="da_spill2":
    ylim = [-1,12]
  # s, ylim = "da_pour", [-1,100]
  # s, ylim = "da_spill2", [-1,12]_
  # s, ylim = ".r", [-1,0]
  # skill = "ti"
  skill = "sa"
  # skill = "selected"
  start = 0

  if False:
    x = np.linspace(start,len(trues[vis_state])-1,len(trues[vis_state])-start)
    fig = plt.figure(figsize=(20,4))
    if s=="da_pour": plt.hlines(0.3,start,len(trues[vis_state])-1,color="red",linestyles="dashed")
    plt.scatter(x, trues[vis_state][start:], c="blue", s=5)
    plt.errorbar(x=x, y=ests[skill][vis_state]["mean"][start:], yerr=ests[skill][vis_state]["sdv"][start:], fmt='o', markersize=3, capsize=2, c="orange", ecolor='pink', zorder=-1)
    plt.xlabel("episode")
    plt.ylabel("E["+s+"] +/- Std["+s+"]")
    # plt.ylim(ylim[0], ylim[1])
    colors = ['blue', 'orange']
    lines = [Line2D([0], [0], color=c, marker='o') for c in colors]
    labels = ['true', 'estimation']
    plt.legend(lines, labels)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, hspace=0.6)
    plt.show()

  if False:
    miss = []
    for i in range(start,len(trues[vis_state])):
      if trues[vis_state][i]<0.3:
        miss.append(envs["smsz"][i])
    fig = plt.figure()
    plt.title("appearance count of da_pour<0.3 ")
    plt.hist(miss, bins=20)
    plt.xlabel("smsz")
    plt.ylabel("count")
    plt.show()

  if True:
    # smsz_list = [0.03, 0.05, 0.07, 0.08]
    smsz_list = [0.03, 0.08]
    for i_s in range(len(smsz_list)-1):
      fig = plt.figure(figsize=(13,4))
      ax = fig.add_subplot(1, 1, 1)
      # ax.set_title(str(smsz_list[i_s])+" ~ "+str(smsz_list[i_s+1]))
      if vis_state=="da_pour": 
        target = 0.3/0.0055
        unit = 1/0.0055
      elif vis_state=="da_spill2": 
        target = 0
        unit = 10
      trues[vis_state] = np.array(trues[vis_state])*unit
      ests[skill][vis_state]["mean"] = np.array(ests[skill][vis_state]["mean"])*unit
      ests[skill][vis_state]["sdv"] = np.array(ests[skill][vis_state]["sdv"])*unit
      if color_list==None: color_list = ["blue"]*len(trues[vis_state])
      for i in range(start,len(trues[vis_state])):
        if smsz_list[i_s]<envs["smsz"][i]<=smsz_list[i_s+1]:
          # c = "blue"
          c = color_list[i]
          # if ests[skill]["skill"]["mean"][i]==0:
          #   c = "green"
          # else:
          #   c = "red"
          ax.scatter(i, trues[vis_state][i], c=c, s=5, zorder=2)
          ax.errorbar(x=i, y=ests[skill][vis_state]["mean"][i], yerr=ests[skill][vis_state]["sdv"][i], fmt='o', markersize=3, capsize=2, c="orange", ecolor='orange', zorder=1)
          # if s=="da_pour" and trues[vis_state][i]<0.3:
          #   # a = "da_total: "+str(trues["da_total"][i])+"\n"+"da_spill2: "+str(trues["da_spill2"][i])
          #   a = "total: "+str(trues["da_total"][i])+"\n"+"spill: "+str(trues["da_spill2"][i])
          #   plt.annotate(a, (i, trues[vis_state][i]), size=7)
          #   Print(i, envs["smsz"][i], trues["da_pour"][i], trues["da_spill2"][i], trues["da_total"][i], trues["p_pour_trg"][i])
      plt.hlines(target,start,len(trues[vis_state])-1,color="lime",linestyles="dashed",zorder=-10)
      ax.set_xlabel("episode")
      label = "da_spill" if vis_state=="da_spill2" else "da_pour"
      # ax.set_ylabel("E["+s+"] +/- Std["+s+"]")
      ax.set_ylabel(label)
      ax.set_ylim(ylim[0], ylim[1])
      colors = [
        'blue',
        # "green",
        "red"
        ]
      # circles = [plt.plot([0], [0], color=c, marker='o', ms=10) for c in colors]
      circles = [plt.plot([],[], marker="o", ls="", mec=None, color=c)[0] for c in colors]
      lines = [
        Line2D([0], [0], color="orange", marker='o'),
        Line2D([0], [0], color="lime", linestyle="dashed")
        ]
      labels = [
        'observed value',
        'observed value (bad or need to check)', 
        # "observed value (Tipping)",
        # "observed value (Shaking)",
        'estimated value (mean +/- 1SD)', 
        "target amount = "+str(int(target))
        ]
      loc = 'lower right' if s=="da_pour" else "upper right"
      ax.legend(
        circles+lines, labels,
        # loc=loc,
        # bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0
        loc = "best"
      )
      ax.set_xlim(0,len(trues[vis_state]))
      ax.set_xticks(np.arange(0, len(trues[vis_state])+1, 50))
      # ax.set_xticks(np.arange(0, len(trues[vis_state])+1, 1), minor=True)
      # ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
      # ax.grid(which='major', alpha=0.9, linestyle='dotted') 
      ax.grid(which='major', alpha=0.9, linestyle='dotted', axis="y") 
      plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, hspace=0.6)
      plt.show()
      