import pandas as pd
import numpy as np
from scipy.stats import iqr
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  target_dir_list = [
    "mtr_sms/infer/basic", 
    # "mtr_sms/infer/additional_early", 
    # "mtr_sms/infer/additional_more", 
    # "mtr_sms/infer/normal_early", 
    # "mtr_sms/infer/normal_more",
    "mtr_sms/infer/additional2_early", 
    "mtr_sms/infer/additional2_more", 
  ]
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  skill_list = ["std_pour","shake_A"]
  mtr_list = ["bounce","nobounce","natto","ketchup"]
  non_viscous_list = ["bounce","nobounce"]
  viscous_list = ["natto","ketchup"]
  sms_list = ["002","0055","009"]

  df_reward = pd.DataFrame(columns=[
    "skill","case","mtr","sms","trial",
    "reward","estimation","abs difference","abs difference(%)"
  ])
  df_reward_organize = pd.DataFrame(columns=[
    "skill","mtr","sms","case",
    "mean reward","mean estimation",
    "mean abs difference","mean abs difference(%)",
    "iqr reward", "iqr estimation",
    "trials"
  ])
  df_reward_organize_macro = pd.DataFrame(columns=[
    "skill","mtr","case",
    "total mean reward","total mean estimation",
    "total mean abs difference","total mean abs difference(%)",
    "total iqr reward", "total iqr estimation"
  ])
  for target_dir in target_dir_list:
    for skill in skill_list:
      total_reward_list = []
      total_estimation_list = []
      total_abs_difference_list = []
      total_non_viscous_reward_list = []
      total_non_viscous_estimation_list = []
      total_non_viscous_abs_difference_list = []
      total_viscous_reward_list = []
      total_viscous_estimation_list = []
      total_viscous_abs_difference_list = []
      for mtr in mtr_list:
        for sms in sms_list:    
          reward_list = []
          estimation_list = []
          abs_difference_list = []

          case = target_dir.split("/")[-1]    
          log_path = root_path + target_dir + "/" \
                    + skill + "/" + mtr+"_"+sms + "/dpl_est.dat"

          data = np.loadtxt(log_path, comments='!')
          for i in range(len(data)):
            reward_list.append(data[i,1])
            estimation_list.append(data[i,2])
            abs_difference_list.append(abs(data[i,1]-data[i,2]))
            df_reward = df_reward.append({
              "case": case,
              "skill": skill,  
              "mtr": mtr, 
              "sms": sms, 
              "trial": i+1, 
              "reward": reward_list[-1], 
              "estimation": estimation_list[-1], 
              "abs difference": abs_difference_list[-1], 
              "abs difference(%)": abs_difference_list[-1]/abs(estimation_list[-1])*100
            }, ignore_index=True)
          
          trials = data[-1][0]+1
          mean_reward = np.mean(reward_list)
          mean_estimation = np.mean(estimation_list)
          mean_abs_difference = np.mean(abs_difference_list)
          mean_abs_difference_p = mean_abs_difference/abs(mean_estimation)*100
          iqr_reward = iqr(reward_list)
          iqr_estimation = iqr(estimation_list)
          df_reward_organize = df_reward_organize.append({
            "case": case,
            "skill": skill,  
            "mtr": mtr, 
            "sms": sms, 
            "trials": trials, 
            "mean reward": mean_reward, 
            "mean estimation": mean_estimation, 
            "mean abs difference": mean_abs_difference, 
            "mean abs difference(%)": mean_abs_difference_p, 
            "iqr reward": iqr_reward, 
            "iqr estimation": iqr_estimation
          }, ignore_index=True)
          total_reward_list += reward_list
          total_estimation_list += estimation_list
          total_abs_difference_list += abs_difference_list
          if mtr in non_viscous_list:
            total_non_viscous_reward_list += reward_list
            total_non_viscous_estimation_list += estimation_list
            total_non_viscous_abs_difference_list += abs_difference_list
          elif mtr in viscous_list:
            total_viscous_reward_list += reward_list
            total_viscous_estimation_list += estimation_list
            total_viscous_abs_difference_list += abs_difference_list

      macro_mtr_list = ["all","non viscous","high viscous"]
      macro_reward_list = [total_reward_list,
                          total_non_viscous_reward_list,
                          total_viscous_reward_list]
      macro_estimation_list = [total_estimation_list,
                              total_non_viscous_estimation_list,
                              total_viscous_estimation_list]
      macro_abs_difference_list = [total_abs_difference_list,
                                  total_non_viscous_abs_difference_list,
                                  total_viscous_abs_difference_list]
      for mtr,reward_list,estimation_list,abs_difference_list in zip(macro_mtr_list,macro_reward_list,macro_estimation_list,macro_abs_difference_list):
        total_mean_reward = np.mean(reward_list)
        total_mean_estimation = np.mean(estimation_list)
        total_mean_abs_difference = np.mean(abs_difference_list)
        total_mean_abs_difference_p = total_mean_abs_difference/abs(total_mean_estimation)*100
        total_iqr_reward = iqr(reward_list)
        total_iqr_estimation = iqr(estimation_list)
        df_reward_organize_macro = df_reward_organize_macro.append({
          "case": case,
          "mtr": mtr, 
          "skill": skill,  
          "total mean reward": total_mean_reward, 
          "total mean estimation": total_mean_estimation, 
          "total mean abs difference": total_mean_abs_difference, 
          "total mean abs difference(%)": total_mean_abs_difference_p, 
          "total iqr reward": total_iqr_reward, 
          "total iqr estimation": total_iqr_estimation
        }, ignore_index=True)

  df_reward.to_csv(
    root_path+"mtr_sms/infer/reward.csv", 
    index=False)
  df_reward_organize = df_reward_organize.sort_values(["skill","mtr","sms"])
  df_reward_organize.to_csv(
    root_path+"mtr_sms/infer/reward_organized.csv", 
    index=False)
  df_reward_organize_macro = df_reward_organize_macro.sort_values(["skill","mtr"])
  df_reward_organize_macro.to_csv(
    root_path+"mtr_sms/infer/reward_organized_macro.csv", 
    index=False)