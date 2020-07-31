import pandas as pd
import numpy as np
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  target_dir_list = [
    "mtr_sms/learn/basic", 
    "mtr_sms/learn/additional_early",
    "mtr_sms/learn/additional_more", 
    "mtr_sms/learn/normal_early", 
    "mtr_sms/learn/normal_more"
  ]
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  dynamics_list = ["Fgrasp","Fmvtorcv_rcvmv","Fmvtorcv","Fmvtopour2",
                    "Fflowc_tip10","Fflowc_shakeA10","Famount4"]
  code_list = ["mean","err"]

  df_model_loss = pd.DataFrame(columns=["case","dynamics","first_loss","final_loss","improve(%)","epochs"])
  for i, dynamics in enumerate(dynamics_list):
    for j, code in enumerate(code_list):
      for target_dir in target_dir_list:
        case = target_dir.split("/")[-1]    
        log_path = root_path + target_dir + "/models/train/" \
                    + "nn_log-"+dynamics+code+".dat"
        if not os.path.exists(log_path):
          time = 0
          while True:
            tmp_log_path = root_path + target_dir + "/models/train/" \
                        + "nn_log-"+"0"*(5-len(str(time)))+str(time)+"-"+dynamics+code+".dat"
            if os.path.exists(tmp_log_path): 
              time += 1
              log_path = tmp_log_path
            else:
              break
        data = np.loadtxt(log_path, comments='!')
        first_loss = data[0,2]
        last_loss = data[-1,2]
        improve = (first_loss-last_loss)/first_loss*100
        epochs = data[-1,0]
        df_model_loss = df_model_loss.append({
          "case": case, 
          "dynamics": dynamics+" "+code,
          "first_loss": first_loss,
          "final_loss": last_loss,  
          "improve(%)": improve, 
          "epochs": epochs
        }, ignore_index=True)

  df_model_loss.set_index(["dynamics"], inplace=True)

  save_path = root_path+"mtr_sms/learn/organize_model_loss.csv"
  df_model_loss.to_csv(save_path)