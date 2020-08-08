import pandas as pd
import numpy as np
from scipy.stats import iqr
from matplotlib import pyplot as plt
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  is_save = False
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
  # root_path = "/tmp/"
  skill_list = ["std_pour","shake_A"]
  # skill_list = ["shake_A"]
  mtr_list = ["bounce","nobounce","natto","ketchup"]
  # mtr_list = ["bounce","nobounce"]
  non_viscous_list = ["bounce","nobounce"]
  viscous_list = ["natto","ketchup"]
  sms_list = ["002","0055","009"]

  df_reward = pd.DataFrame(columns=[
    "skill","case","mtr","sms","trial",
    "reward","estimation","abs difference","abs difference(%)"
  ])
  df_reward_organize = pd.DataFrame(columns=[
    "skill","mtr","sms","case",
    "mean reward","std reward","mean estimation","std estimation",
    "mean abs difference","std abs difference","mean abs difference(%)",
    "trials"
  ])
  df_reward_organize_macro = pd.DataFrame(columns=[
    "skill","mtr","case",
    "mean reward","std reward","mean estimation","std estimation",
    "mean abs difference","std abs difference","mean abs difference(%)",
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
          std_reward = np.std(reward_list)
          std_estimation = np.std(estimation_list)
          std_abs_difference = np.std(abs_difference_list)
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
            "std reward": std_reward, 
            "std estimation": std_estimation,
            "std abs difference": std_abs_difference
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
        total_std_reward = np.std(reward_list)
        total_std_estimation = np.std(estimation_list)
        total_std_abs_difference = np.std(abs_difference_list)
        df_reward_organize_macro = df_reward_organize_macro.append({
          "case": case,
          "mtr": mtr, 
          "skill": skill,  
          "mean reward": total_mean_reward, 
          "mean estimation": total_mean_estimation, 
          "mean abs difference": total_mean_abs_difference, 
          "mean abs difference(%)": total_mean_abs_difference_p, 
          "std reward": total_std_reward, 
          "std estimation": total_std_estimation,
          "std abs difference": total_std_abs_difference
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

  fig, ax = plt.subplots(figsize=(20,7))
  ax.grid(c="gray",zorder=0,linewidth=0.5)
  x = np.arange(0,len(df_reward_organize_macro))
  x_ticklabels = [row["skill"]+"\n"+row["mtr"]+"\n"+row["case"]
                  for index,row in df_reward_organize_macro.iterrows()]
  ax.bar(
    x, 
    df_reward_organize_macro["mean abs difference"],
    yerr=df_reward_organize_macro["std abs difference"],
    width=0.3,
  )
  ax.axhline(y=0, xmin=0, xmax=len(df_reward_organize_macro),c="black",linewidth=0.5)
  ax.set_title("mean +/- std abs difference each skill/mtr/type")
  ax.set_ylabel("mean +/- std abs difference")
  ax.set_xlabel("skill / mtr / type",fontsize=10)
  ax.set_xticks(x)
  ax.set_xticklabels(x_ticklabels,fontsize=8)
  fig.subplots_adjust(left=0.05,right=0.98,bottom=0.13,top=0.87)
  if is_save:
    fig.savefig("/home/yashima/Pictures/mtr_sms/reward/reward_organized_macro.png")
  # fig.show()
  plt.close()


  skill_list = ["shake_A","std_pour"]
  mtr_set_list = [["bounce","nobounce"],["natto","ketchup"]]
  mtr_type_list = ["non_viscous","high_viscous"]
  for skill in skill_list:
    for mtr_set,mtr_type in zip(mtr_set_list,mtr_type_list):
      df = df_reward_organize[((df_reward_organize["mtr"]==mtr_set[0]) | (df_reward_organize["mtr"]==mtr_set[1])) & 
                              (df_reward_organize["skill"]==skill)]
      fig, ax = plt.subplots(figsize=(20,7))
      ax.grid(c="gray",zorder=0,linewidth=0.5)
      x = np.arange(0,len(df))
      x_ticklabels = [row["mtr"]+"_"+str(row["sms"])+"\n"+row["case"]
                      for index,row in df.iterrows()]
      ax.bar(
        x, 
        df["mean abs difference"],
        yerr=df["std abs difference"],
        width=0.3,
      )
      ax.axhline(y=0, xmin=0, xmax=len(df),c="black",linewidth=0.5)
      ax.set_title("mean +/- std abs difference each sms/type of {}/{}".format(skill,mtr_type))
      ax.set_ylabel("mean +/- std abs difference")
      ax.set_xlabel("sms / type",fontsize=10)
      ax.set_ylim(0,8)
      ax.set_xticks(x)
      ax.set_xticklabels(x_ticklabels,fontsize=8)
      fig.subplots_adjust(left=0.05,right=0.98,bottom=0.13,top=0.87)
      if is_save:
        fig.savefig("/home/yashima/Pictures/mtr_sms/reward/{}_{}.png".format(skill,mtr_type))
      # fig.show()
      # plt.close()