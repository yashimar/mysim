import joblib
import matplotlib.pyplot as plt
import numpy as np
from core_tool import *

def Help():
  pass

def Run(ct, *args):
  ep  = 6
  node_est_from = ("n2a", 2)    #for read tree, first term is planning node, second term is repeat time
  node_diff_XS = ("n3sa", 1)    #for read db, first term is validate node, second term is repeat time-1 (list index)
  file_path = "mtr_sms_sv/learn/shake_A/random/0055/normal/"
  
  root_dir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  if node_est_from[1]==None: tree_name = "ep"+str(ep)+"_"+node_est_from[0]+".jb"
  else: tree_name = "ep"+str(ep)+"_"+node_est_from[0]+"_"+str(node_est_from[1])+".jb"
  full_xs = LoadYAML(root_dir+file_path+"database.yaml")['Entry'][ep]['Seq']
  s = map(lambda x: x["Name"], full_xs)
  i_node = [i for i,x in enumerate(s) if x==node_diff_XS[0]][node_diff_XS[1]]
  db_xs = full_xs[i_node]["XS"]
    
  tree = joblib.load(root_dir+file_path+"best_est_trees/"+tree_name, "r")
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
  Print("-"*50)

  pair_keys = TPair(node_diff_XS[0], 0)   #ptree exists each repeat time, so second argument should be 0
  Print("estimate",node_diff_XS,"XS:\n",tree.Tree[pair_keys].XS)
  print("")
  Print("actual",node_diff_XS,"XS:\n",db_xs)