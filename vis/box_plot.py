import matplotlib.pyplot as plt
import pandas as pd
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  skill = args[0]
  target_file = "mtr_sms/infer/reward.csv"
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  df = pd.read_csv(root_path+target_file)

  skill_list = [skill]
  case_list = ["basic","additional2_early","additional2_more"]
  mtr_list = ["bounce","nobounce","natto","ketchup"]
  sms_list = [2,55,9]

  fig = plt.figure(figsize=(20,12))
  fig.suptitle(skill_list[0])
  for skill in skill_list:
    for k,mtr in enumerate(mtr_list):
      for j,sms in enumerate(sms_list):
        reward_case = []
        for case in case_list:
          data = df[(df["skill"]==skill) & 
                    (df["case"]==case) & 
                    (df["mtr"]==mtr) &
                    (df["sms"]==sms)]
          reward = data["reward"]
          reward_case.append(reward)
        ax = fig.add_subplot(len(sms_list),len(mtr_list),len(mtr_list)*j+k+1)
        ax.set_title(mtr+"_"+str(sms),fontsize=8)
        ax.boxplot(reward_case,labels=case_list)