import yaml
import numpy as np
import pickle
from core_tool import *

def Help():
  pass

def Run(ct, *args):
  name = args[0]
  database_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" \
                  + name + "/database.yaml"
  with open(database_path) as f:
    database = yaml.safe_load(f)

  eps_list = []
  for i in range(len(database["Entry"])):
    j = 0
    keys = []
    rewards = []
    reward = database["Entry"][i]["R"]
    reward_counter = 1
    while True:
      try:
        data = database["Entry"][i]["Seq"][j]["XS"]
        # print(data)
        state = database["Entry"][i]["Seq"][j]["Name"]
        # print(state)
        if j==0: 
          rewards.append({"total":reward})
          # keys.append(reward)
          keys.append({
                        "a_trg": data["a_trg"]["X"], 
                        "ps_rcv": data["ps_rcv"]["X"], 
                        "size_srcmouth": data["size_srcmouth"]["X"], 
                        "material2": data["material2"]["X"]})
        if state=="n0": keys.append({"gh_ratio": data["gh_ratio"]["X"]})
        if state=="n2a": keys.append({"p_pour_trg": data["p_pour_trg"]["X"]})
        if state=="n3ti": keys.append({"dtheta1": data["dtheta1"]["X"],
                                      "dtheta2": data["dtheta2"]["X"]})
        if state=="n3sa": keys.append({"dtheta1": data["dtheta1"]["X"],
                                      "shake_spd": data["shake_spd"]["X"],
                                      "shake_axis2": data["shake_axis2"]["X"]})
        if state=="n1rcvmvr": 
          rewards.append({str(reward_counter)+"_n1rcvmvr": data[".r"]["X"][0][0]})
          reward_counter += 1
        if state=="n2br": 
          rewards.append({str(reward_counter)+"_n2br": data[".r"]["X"][0][0]})
          reward_counter += 1
        if state=="n4tir": 
          rewards.append([{str(reward_counter)+"_n4tir": data[".r"]["X"][0][0]},
                          [{"da_trg": data["da_trg"]["X"][0][0]}], 
                          [{"da_spill2": data["da_spill2"]["X"][0][0]}], 
                          [{"da_pour": data["da_pour"]["X"][0][0]}], 
                          [{"a_pour": data["a_pour"]["X"][0][0]}], 
                          [{"a_spill2": data["a_spill2"]["X"][0][0]}]])
          reward_counter += 1
        if state=="n4sar": 
          rewards.append([{str(reward_counter)+"_n4sar": data[".r"]["X"][0][0]}, 
                          [{"da_trg": data["da_trg"]["X"][0][0]}], 
                          [{"da_spill2": data["da_spill2"]["X"][0][0]}], 
                          [{"da_pour": data["da_pour"]["X"][0][0]}], 
                          [{"a_pour": data["a_pour"]["X"][0][0]}], 
                          [{"a_spill2": data["a_spill2"]["X"][0][0]}]])
          reward_counter += 1
      except:
        state = "end"
        break
      finally:
        j += 1
    keys.insert(0, rewards)
    eps_list.append(keys)
  # print(eps_list)

  save_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  # if = open(save_path+skill+str(Num)+"_sequence_list.txt", "wb")
  # pickle.dump(eps_list, f)
  # f = open(save_path+"choose_skill"+str(Num)+"_sequence_list.txt", "rb")
  # eps_list = pickle.load(f)
      
  with open(save_path+name+"_sequence_list.yaml", "wb") as f:
    for i, eps in enumerate(eps_list):
      yaml.dump({
        i: {"reward": eps[0], 
            "config": eps[1], 
            "sequence": {j: state for j, state in enumerate(eps[2:])}}
      }, f, default_flow_style=False)
      