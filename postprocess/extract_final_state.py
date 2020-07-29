import pandas as pd
import numpy as np
import yaml
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  target_dir = args[0]
  skill_list = ["choose"]
  mtr_list = ["bounce","nobounce","natto","ketchup"]
  sms_list = ["002","0055","009"]
  log_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"+target_dir+"/"
  df_final_state = pd.DataFrame(columns=["a_pour","a_spill2"])

  for skill in skill_list:
    for mtr in mtr_list:
      for sms in sms_list:
        data_path = log_path+skill+"/"+mtr+"_"+sms+"_sequence_list.yaml"
        CPrint(3,data_path)
        with open(data_path, 'r') as yml:
          data = yaml.load(yml)
          for i in range(5):
            a_pour = data[i]["reward"][-1][-2][0]["a_pour"]
            a_spill2 = data[i]["reward"][-1][-1][0]["a_spill2"]
            df_final_state = df_final_state.append({
              "skill": skill, 
              "mtr": mtr, 
              "sms": sms, 
              # "num": i, 
              "a_pour": a_pour, 
              "a_spill2": a_spill2
            }, ignore_index=True)

  df_final_state.set_index(["skill","mtr","sms"], inplace=True)
  df_final_state.to_csv(log_path+"final_state.csv")
