import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from core_tool import *

def Help():
  pass

def Run(ct, *args):
  # only follow not repeat
  ### arguments
  # ep_list = [199]
  ep_start = 0
  ep_end = 19
  ep_list = np.linspace(0,ep_end,ep_end-ep_start+1).astype(int)
  node_est_from = ("n0", None)    #for read tree, first term is planning node, second term is repeat time
  file_path = "mtr_sms_sv/validation/shake_A/natto/0055/20search_nodb/"
  obs_keys = [
    #(key, observe node)
    # ("gh_abs",    "n1"),      #Fgrasp
    # ("ps_rcv",    "n2a"),     #Fmvtorcv
    # ("p_pour",    "n2a"),     #Fmvtorcv
    # ("dps_rcv",   "n1rcvmv"), #Fmvtorcv_rcvmv
    # ("v_rcv",     "n1rcvmv"), #Fmvtorcv_rcvmv
    # ("lp_pour",   "n2b"),     #Fmvtopour2
    ("da_total",  "n3sa"),    #Fflowc_shakeA10
    ("lp_flow",   "n3sa"),    #Fflowc_shakeA10
    ("flow_var",  "n3sa"),    #Fflowc_shakeA10
    ("da_pour",   "n4sa"),    #Famount4
    ("da_spill2", "n4sa")     #Famount4
  ]
  ###
  root_dir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  base_db = LoadYAML(root_dir+file_path+"database.yaml")['Entry']

  loss_contaier = defaultdict()
  for ep in ep_list:
    if node_est_from[1]==None: tree_name = "ep"+str(ep)+"_"+node_est_from[0]+".jb"
    else: tree_name = "ep"+str(ep)+"_"+node_est_from[0]+"_"+str(node_est_from[1])+".jb"
    
    tree = joblib.load(root_dir+file_path+"best_est_trees/"+tree_name, "r")
    db = base_db[ep]['Seq']
    node_list = map(lambda x: x["Name"], db)

    
    CPrint(2,"="*20,"ep",ep,"="*20)
    # Print("Start:",tree.Start)              #The key of a start node (a key of self.Tree)
    # Print("Tree:",tree.Tree)                #{key:node,...}, key is TPair(key_graph,num_visits), node is a TPlanningNode
    # #Note: key_graph is a key of TGraphDynDomain.Graph (str), num_visits is number of visits (start from 0).
    # Print("Terminal:",tree.Terminal)        #Terminal nodes (a list of keys of self.Tree)
    # Print("BwdOrder:",tree.BwdOrder)        #Order of backward computation (a list of keys of self.Tree)
    # Print("Actions:",tree.Actions)          #[key_xssa,...], actions to be planned, key_xssa is a key of XSSA
    # Print("Selections:",tree.Selections)    #[key_xssa,...], selections to be planned, key_xssa is a key of XSSA
    # Print("Models:",tree.Models)            #[key_F,...], models used, key_F is a key of TGraphDynDomain.Models (str)
    # Print("FlagFwd:",tree.FlagFwd)          #Forward is 0:Not computed, 1:Computed without gradients, 2:Computed with gradients.
    # Print("FlagBwd:",tree.FlagBwd)          #Backward is 0:Not computed, 1:Computed.
    # Print("Value J from start node:",tree.Value())
    # Print("-"*50)

    # pair_keys = TPair(node_diff_XS[0], 0)   #ptree exists each repeat time, so second argument should be 0
    # Print("estimate",node_diff_XS,"XS:\n",tree.Tree[pair_keys].XS)
    # print("")
    # Print("actual",node_diff_XS,"XS:\n",db_xs)
    for key_node in obs_keys:
      print("-"*10)
      key, node = key_node[0], key_node[1]
      node_repeat = TPair(node, 0)   #ptree exists each repeat time, so second argument should be 0
      estX = tree.Tree[node_repeat].XS[key].X
      i_node = [i for i,x in enumerate(node_list) if x==node][0]    #only first reach at each node
      actX = np.matrix(db[i_node]["XS"][key]["X"])
      l2_norm = np.linalg.norm(estX-actX,ord=2)

      Print("estimate",node,key,"X:",estX)
      Print("actual",node,key,"X:",actX)
      Print("l2 norm:", l2_norm)

      if key not in loss_contaier.keys():
        loss_contaier[key] = [l2_norm]
      else:
        loss_contaier[key].append(l2_norm)

  for key_node in obs_keys:
    key = key_node[0]
    plt.figure()
    plt.title(key)
    plt.grid()
    plt.xticks(np.arange(0, len(ep_list), 1))
    plt.plot(loss_contaier[key])
    plt.xlabel("episode")
    plt.ylabel("|| actual - estimate ||")
    plt.show()

  plt.figure(figsize=(6,6))
  diff_list = [loss_contaier[key_node[0]] for key_node in obs_keys]
  labels = [key_node[0]+"\n"
            + str(round(np.mean(diff),3))+"\n"
            +"+/-"+str(round(np.std(diff),3))
            for key_node, diff in zip(obs_keys, diff_list)]
  plt.boxplot(diff_list, labels=labels)
  plt.title(file_path)
  plt.ylabel("|| actual output - estimation ||")

  plt.show()