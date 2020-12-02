from core_tool import *
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib
from scipy.spatial import distance
from scipy.stats import norm
from collections import defaultdict

def Help():  
  pass

def Run(ct, *args):
  name_log = args[0]
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  sl_path = root_path + name_log + "/sequence_list.yaml"

  with open(sl_path, "r") as yml:
    sl = yaml.load(yml)

  d = defaultdict(list)
  for ep in range(len(sl)):
    try:
      sequence = sl[ep]["sequence"]

      d["gh_abs"].append(sequence[1]["n1"]["gh_abs"][0][0])
      d["da_trg"].append(sl[ep]["reward"][3][1][0]["da_trg"])
      d["lp_pour_x"].append(sequence[3]["n2b"]["lp_pour"][0][0])
      d["lp_pour_z"].append(sequence[3]["n2b"]["lp_pour"][2][0])
      d["dtheta1"].append(sequence[4]["n3sa"]["dtheta1"][0][0])
      d["shake_spd"].append(sequence[4]["n3sa"]["shake_spd"][0][0])
      d["shake_axis2_range"].append(sequence[4]["n3sa"]["shake_axis2"][0][0])
      d["shake_axis2_angle"].append(sequence[4]["n3sa"]["shake_axis2"][1][0])

      d["da_total"].append(sequence[5]["n4sar"]["da_total"][0][0])
      d["lp_flow_x"].append(sequence[5]["n4sar"]["lp_flow"][0][0])
      d["lp_flow_y"].append(sequence[5]["n4sar"]["lp_flow"][1][0])
      d["flow_var"].append(sequence[5]["n4sar"]["flow_var"][0][0])
    except:
      pass

  inputs = ["gh_abs", "da_trg", "lp_pour_x", "lp_pour_z", "dtheta1", "shake_spd", "shake_axis2_range", "shake_axis2_angle"]
  outputs = ["da_total", "lp_flow_x", "lp_flow_y", "flow_var"]

  for in_value in inputs:
    fig = plt.figure(figsize=(20,5))
    for i, out_value in enumerate(outputs):
      fig.add_subplot(1, len(outputs), i+1).scatter(x=d[in_value], y=d[out_value])
      plt.title("Correlation coefficient: "+str(round(np.corrcoef(d[in_value], d[out_value])[0][1], 2)))
      plt.xlabel(in_value)
      plt.ylabel(out_value)
      plt.legend()
    plt.show()
  