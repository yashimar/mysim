import pandas as pd
import numpy as np
import yaml
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  skill_list = ["std_pour","shake_A","choose"]
  mtr_list = ["bounce","nobounce","natto","ketchup"]
  sms_list = ["002","0055","009"]
  log_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/try_dpl/"
  df_init_skill_params = pd.DataFrame(columns=["skill","mtr","sms","num","gh_ratio","p_pour_trg","dtheta1","dtheta2","shake_axis2","shake_spd"])

  for skill in skill_list:
    for mtr in mtr_list:
      for sms in sms_list:
        data_path = log_path+skill+"/"+mtr+"_"+sms+"_sequence_list.yaml"
        CPrint(3,data_path)
        with open(data_path, 'r') as yml:
          data = yaml.load(yml)
          for i in range(5):
            gh_ratio = data[i]["sequence"][0]["gh_ratio"][0][0]
            p_pour_trg = [data[i]["sequence"][1]["p_pour_trg"][0][0],
                          data[i]["sequence"][1]["p_pour_trg"][1][0]]
            dtheta1 = data[i]["sequence"][2]["dtheta1"][0][0]
            dtheta2 = data[i]["sequence"][2]["dtheta2"][0][0] if "dtheta2" in data[i]["sequence"][2] else np.nan
            shake_axis2 = [data[i]["sequence"][2]["shake_axis2"][0][0], 
                           data[i]["sequence"][2]["shake_axis2"][1][0]] if "shake_axis2" in data[i]["sequence"][2] else np.nan
            shake_spd = data[i]["sequence"][2]["shake_spd"][0][0] if "shake_spd" in data[i]["sequence"][2] else np.nan
            df_init_skill_params = df_init_skill_params.append({
              "skill": skill, 
              "mtr": mtr, 
              "sms": sms, 
              "num": i, 
              "gh_ratio": gh_ratio, 
              "p_pour_trg": p_pour_trg, 
              "dtheta1": dtheta1, 
              "dtheta2": dtheta2, 
              "shake_axis2": shake_axis2, 
              "shake_spd": shake_spd
            }, ignore_index=True)

  df_init_skill_params.set_index(["skill","mtr","sms"], inplace=True)
  df_init_skill_params.to_csv(log_path+"init_skill_params.csv")
