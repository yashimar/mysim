from core_tool import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn import cluster, preprocessing, mixture
from ..load_history import load_db, get_state_histories, make_sa_df

def Help():
  pass

def create_best_gmm(X, plot_bic=True):
  lowest_bic = np.Infinity
  bic_list = []
  n_components_range = range(1, 11)
  cv_types = ['spherical', 'tied', 'diag', 'full']
  # cv_types = ['full']
  for cv_type in cv_types:
    for n_components in n_components_range:
      gmm = mixture.GaussianMixture(
        n_components=n_components, 
        covariance_type=cv_type,
        # max_iter=300,
      )
      gmm.fit(X)
      bic = gmm.bic(X)
      if bic < lowest_bic:
        lowest_bic = bic
        best_gmm = gmm
        best_cv_type = None
      bic_list.append(bic)

  if plot_bic:
    bic = np.array(bic_list)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    bars = []

    plt.figure(figsize=(15,4))
    spl = plt.subplot(1, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
      xpos = np.array(n_components_range) + .2 * (i - 2)
      bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                    (i + 1) * len(n_components_range)],
                          width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    plt.show()

  return best_gmm

# def marginalize():
  

def Run(ct, *args):
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  log_name = "debugger/ex5/shake_A/natto/random/first"
  init_states_node_unit_pair = [
    ["size_srcmouth", ["n0"], 1],
    # ["material2", ["n0"], 1],
  ]
  skill_params_node_unit_pair = [
    ["p_pour_trg", ["n0"], 1],
    # ["dtheta1", ["n0"], 1],
    # ["dtheta2", ["n4tir"], 1],
    ["shake_spd", ["n0"], 1],
    ["shake_axis2", ["n0"], 1],
    # ["skill", ["n0"], 1]
  ]
  input_sa_node_unit_pair = init_states_node_unit_pair + skill_params_node_unit_pair
  db = load_db(root_path,log_name)
  state_histories = get_state_histories(db, input_sa_node_unit_pair)
  input_sa_df = make_sa_df(state_histories, input_sa_node_unit_pair)
  sc = preprocessing.StandardScaler()
  sc.fit(input_sa_df)
  X = sc.transform(input_sa_df)

  gmm = create_best_gmm(X, plot_bic=True)

  print(gmm.predict_proba(X))
