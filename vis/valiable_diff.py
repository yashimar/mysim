import joblib
import matplotlib.pyplot as plt
import numpy as np
from core_tool import *

def Help():
  pass

def Run(ct, *args):
  ep  = 3
  node_est_from = "n0"
  repeat_time = ""   #for n2a repeat, example _1 (for n2a_1)
  node_diff_XS = "n1"
  file_path = "mtr_sms_sv/learn/shake_A/random/0055/normal/best_est_trees/"
  
  root_dir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  tree_name = "ep"+str(ep)+"_"+node_est_from+repeat_time+".jb"
  db = LoadYAML(root_dir+file_path+"database.yaml")['Entry'][ep]['Seq'][node_diff_XS]
  tree = joblib.load(root_dir+file_path+tree_name, "r")
  Print("Start:",tree.Start)              #The key of a start node (a key of self.Tree)
  Print("Tree:",tree.Tree)                #{key:node,...}, key is TPair(key_graph,num_visits), node is a TPlanningNode
  #Note: key_graph is a key of TGraphDynDomain.Graph (str), num_visits is number of visits (start from 0).
  Print("Terminal:",tree.Terminal)        #Terminal nodes (a list of keys of self.Tree)
  Print("BwdOrder:",tree.BwdOrder)        #Order of backward computation (a list of keys of self.Tree)
  Print("Actions:",tree.Actions)          #[key_xssa,...], actions to be planned, key_xssa is a key of XSSA
  Print("Selections:",tree.Selections)    #[key_xssa,...], selections to be planned, key_xssa is a key of XSSA
  Print("Models:",tree.Models)            #[key_F,...], models used, key_F is a key of TGraphDynDomain.Models (str)
  Print("FlagFwd:",tree.FlagFwd)          #Forward is 0:Not computed, 1:Computed without gradients, 2:Computed with gradients.
  Print("FlagBwd:",tree.FlagBwd)          #Backward is 0:Not computed, 1:Computed.
  Print("Value J from start node:",tree.Value())

  for pair_keys in tree.Tree.keys():
    key = pair_keys.A
    if key==node_diff_XS:
      Print("estimate",node_diff_XS,"XS:",tree.Tree[pair_keys].XS)