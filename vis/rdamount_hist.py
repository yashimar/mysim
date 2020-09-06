import numpy as np
from matplotlib import pyplot as plt
from core_tool import *
import yaml
import glob
from collections import defaultdict, Counter
import pandas as pd

def Help():
  pass

def Run(ct,*args):
  i_episode = 7
  i_node_list = [0,1]
  # validation_keys = ["a_total"]
  validation_keys = None
  # log_name = args[0]
  # root_logpath = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/replay_log/" \
  #               + "mtr_sms/infer/"+log_name+"/"
  root_logpath = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/" \
                + "debug/replay/use_sv/"
  data_list_dict = defaultdict()

  da_pour_list = []
  da_spill2_list = []
  rdamount_list = []
  data_list = []
  file_list = glob.glob(root_logpath+"*_ep"+str(i_episode)+"_*")
  
  for i_node in i_node_list:
    data_list_dict[i_node] = defaultdict() 

  for file_path in file_list:
    fp = open(file_path,"r")
    data_main = yaml.load(fp)
    fp.close()
    for i_node in i_node_list:
      data = data_main[i_node]
      for key in data.keys():
        if key not in data_list_dict[i_node].keys():
          data_list_dict[i_node][key] = defaultdict()
        for i in range(len(data[key]["X"])):
          if i not in data_list_dict[i_node][key].keys():
            data_list_dict[i_node][key][i] = []
          data_list_dict[i_node][key][i].append(data[key]["X"][i][0])
          # if data[key]["X"][i][0]==0.202044633638758:
          #   CPrint(2,file_path)

  for i_node in i_node_list:
    CPrint(1,"="*30,"node:",i_node,"="*30)
    key_list = validation_keys if validation_keys!=None else data_list_dict[i_node].keys()
    for key in key_list:
      CPrint(2,"====== key:",key,"======")
      for i in range(len(data_list_dict[i_node][key])):
        data_list = data_list_dict[i_node][key][i]
        c = Counter(data_list)
        value_range = max(data_list) - min(data_list)
        CPrint(3,"-- component:",i,"--")
        Print("value range:",value_range)
        Print("uniques:",len(c))
        # Print(c)
        # if value_range>1e-3: Print(c)
        # for j, data in enumerate(data_list):
        #   if c[data]==2:
        #     print(j)

  # plt.hist(data_list_dict[3]["Sresume_Fobs3"][0])
  # fig = plt.figure()
  # fig.add_subplot(1,2,1).hist(data_list_dict[3]["Sresume_Fobs3"][0])
  # fig.add_subplot(1,2,2).hist(data_list_dict[4]["Sresume_Fobs4"][0])
  # print(Counter(data_list_dict[1]["lp_pour"][0]))

  # df = pd.DataFrame(columns=["node","key","component","uniques","value_range"])
  # for i_node in i_node_list:
  #   for key in data_list_dict[i_node].keys():
  #     for i in range(len(data_list_dict[i_node][key])):
  #       data_list = data_list_dict[i_node][key][i]
  #       c = Counter(data_list)
  #       value_range = max(data_list) - min(data_list)
  #       df = df.append({
  #         "node": i_node, 
  #         "key": key, 
  #         "component": i, 
  #         "uniques": len(c), 
  #         "value_range": value_range
  #       },ignore_index=True)
  # df.to_csv("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/debug/value_valiation3.csv",index=False)