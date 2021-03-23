from core_tool import *
import yaml
import numpy as np
import pickle
from collections import defaultdict

def load_db(root_path, log_name):
  database_path = root_path + log_name + "/database.yaml"
  Print("Loading database ...")
  with open(database_path) as f:
    database = yaml.safe_load(f)
  Print("successfully loaded")
  return database["Entry"]

def load_est_tree(root_path, log_name, db):
  tree_path = root_path + log_name + "/best_est_trees/"
  ests = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

  i = 0
  while True:
    try:
      with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
        tree = joblib.load(f)
        n_tir, n_sar, skill = None, None, None
        for n in tree.Tree.keys():
          if n.A=="n4tir": n_tir = n, skill = n_tir["skill"].X.item()
          elif n.A=="n4sar": n_sar = n, skill = n_sar["skill"].X.item()
        tir_xs = tree.Tree[n_tir].XS if n_tir is not None else None
        sar_xs = tree.Tree[n_sar].XS if n_sar is not None else None
        selected_xs = tir_xs if skill==0 else sar_xs
        # for (est_dict, r_xs) in zip([ests["sa"]], [sar_xs]):
        # for (est_dict, r_xs) in zip([ests["ti"]], [tir_xs]):
        for (est_dict, r_xs) in zip([ests["ti"], ests["sa"], ests["selected"]], [tir_xs, sar_xs, selected_xs]):
          for s in ["da_spill2", "da_pour", ".r"]:
            est_dict[s]["mean"].append(r_xs[s].X.item())
            est_dict[s]["sdv"].append(np.sqrt(r_xs[s].Cov.item()))
          est_dict["skill"]["mean"].append(r_xs["skill"].X.item())
    except:
      break
  return ests


def format_x(x):
  formated_x = sum(x,[])
  if len(formated_x)==1:
    formated_x = formated_x[0]
  return formated_x

# state_node_unit_pair = [
#     ["da_pour", ["n4tir","n4sar"], 1/0.0055],
#     ["da_spill2", ["n4tir","n4sar"], 10]
#   ]
#   db = load_db(root_path,log_name)["Entry"]
def get_state_histories(db,state_node_unit_pair):
  state_histories = defaultdict(list)
  for state,possible_nodes,unit in  state_node_unit_pair:
    for i in range(len(db)):
      seq = db[i]["Seq"]
      for node_xs in seq:
        node = node_xs["Name"]
        xs = node_xs["XS"]
        if node in possible_nodes:
          state_histories[state].append(format_x(xs[state]["X"]))
    state_histories[state] = np.array(state_histories[state])*unit
  return state_histories