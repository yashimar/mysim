from numpy.lib.function_base import copy
from numpy.lib.scimath import logn
from scipy.spatial.kdtree import distance_matrix
from core_tool import *
import os
from copy import deepcopy
import yaml
import pickle
import numpy as np
from scipy.spatial import distance
from scipy.stats import binom_test
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label
from sklearn.linear_model import Lasso
import pandas as pd
pd.options.display.precision = 4
from matplotlib import pyplot as plt
from matplotlib import ticker
from load_history import load_db, get_state_histories

ROOT_PATH = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
SAVE_REPORT_ROOT_PATH = "/home/yashima/Pictures/debugging/test5/"
SAVE_REPORT_PATH = ""

def Help():
  pass

def save_plt_fig(save_dir, file_name):
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  plt.savefig(save_dir+"/"+file_name+".png")
  plt.close()

def make_bad_sample_idx_list(target_state, history, WINDOW_SIZE=100, IQR_MAGNIFICATION=1.5, return_threshold=-1.0):
  h_q1,h_q2,h_q3 = [[np.percentile(history[target_state][i-WINDOW_SIZE:i], [p]) for i in range(WINDOW_SIZE,len(history[target_state]))] for p in [25,50,75]]
  bad_sample_idx_list = []
  for i, (x,q1,q2,q3) in enumerate(zip(history[target_state][WINDOW_SIZE:],h_q1,h_q2,h_q3)):
    iqr = q3 - q1
    border_min, border_max = q1-iqr*IQR_MAGNIFICATION-1e-2, q3+iqr*IQR_MAGNIFICATION+1e-2
    if x<=border_min or x>=border_max or history[".r"][WINDOW_SIZE+i]<return_threshold:
      bad_sample_idx_list.append(WINDOW_SIZE+i)
  return bad_sample_idx_list

# def make_bad_return_idx_list(all_sa_df, start_idx=100, return_thresold=-1.0):
#   all_sa_df = all_sa_df.iloc[start_idx:]
#   bad_return_idx_list = all_sa_df[all_sa_df[".r"] < return_thresold].index.tolist()
#   return bad_return_idx_list

def make_sa_df(state_histories, sa_node_unit_pair):
  sa_key_list = []
  sa_matrix = []
  for sa_key,_,_ in sa_node_unit_pair:
    sa_list = state_histories[sa_key].tolist()
    if type(sa_list[0])!=list: 
      sa_key_list.append(sa_key)
      sa_list = [sa_list]
    else:
      sa_key_list += [sa_key+"_dim"+str(i+1) for i in range(len(sa_list[0]))]
      sa_list = np.array(sa_list).T.tolist()
    sa_matrix += sa_list
  sa_matrix = np.array(sa_matrix).T
  sa_df = pd.DataFrame(sa_matrix, columns=sa_key_list)
  return sa_df

def make_pca_sa_df(sa_df, do_plot_pca_info=False):
  ss = StandardScaler()
  sa_df_norm = ss.fit_transform(sa_df)
  pca = PCA()
  pca.fit(sa_df_norm)
  feature = pca.transform(sa_df_norm)
  feature_df = pd.DataFrame(
    feature,
    columns=["PC{}".format(x + 1) for x in range(feature.shape[1])]
  )
  pca_components_df = pd.DataFrame(
    pca.components_,
    columns=sa_df.columns,
    index=["PC{}".format(x + 1) for x in range(feature.shape[1])]
  )

  if do_plot_pca_info:
    print(pca_components_df)

    fig, ax = plt.subplots(1,1,figsize=(10,3))
    ax.axis('tight')
    ax.axis('off')
    ax.table(
      cellText=pca_components_df.values.round(2),
      colLabels=pca_components_df.columns,
      loc='center',
      colWidths=[.10]*len(pca_components_df.columns)
      # bbox=[0,0,1,1]
    )
    # plt.show()
    save_plt_fig(SAVE_REPORT_PATH+"pca_reuslt/", "pca_components")

    plt.figure(figsize=(5,5))
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    # plt.show()
    save_plt_fig(SAVE_REPORT_PATH+"pca_reuslt/", "cumulative_contribution")

    # plt.figure(figsize=(5,5))
    # plt.grid()
    # plt.scatter(x=feature[:,0], y=feature[:,1], s=8)
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    ## plt.show()
    # save_plt_fig(SAVE_REPORT_PATH, "pca_2d_scatter")

  return feature_df

def make_nearby_sample_list_from_target_sample(dist_M, target_sample_idx):
  dists_from_target_sample = dist_M[target_sample_idx]
  near_sample_idx_list = dists_from_target_sample.argsort()
  near_sample_distance_list = dists_from_target_sample[near_sample_idx_list]
  return near_sample_idx_list, near_sample_distance_list

def make_nearby_comparsion_df(l, kind, dist_M, target_sample_idx, neighbor_threshold=1.0, n_consider=10):
  nearby_sample_idx_list, nearby_sample_distance_list = make_nearby_sample_list_from_target_sample(dist_M, target_sample_idx)
  nearby_comparsion_df = l.all_sa_df.iloc[nearby_sample_idx_list]
  nearby_comparsion_df.insert(0, "distance", nearby_sample_distance_list)
  nearby_comparsion_df.insert(0, "episode", nearby_sample_idx_list)
  nearby_comparsion_df.reset_index(inplace=True, drop=True)

  nearby_samples_bad_sample_key_list = []
  first_faraway_samples_order = len(nearby_sample_distance_list)

  for order, d in enumerate(nearby_sample_distance_list):
    if d > neighbor_threshold:
      first_faraway_samples_order = order
      break

  x = np.linspace(0,n_consider-1,n_consider)
  colors = ["red"] + ["blue"]*(first_faraway_samples_order-1) + ["gray"]*(n_consider-first_faraway_samples_order)
  for sa_key, sa_series in nearby_comparsion_df.iloc[:,2:].iteritems():
    sa_values = sa_series.values[:n_consider]
    plt.figure(figsize=(10,5))
    plt.bar(x, sa_values, tick_label=nearby_comparsion_df["episode"].values[:n_consider], color=colors)
    save_plt_fig(SAVE_REPORT_PATH+"nearby_comparsion/"+str(target_sample_idx)+"/"+kind+"/",str(sa_key))

    q1,q2,q3 = [np.percentile(sa_values[1:], [p]) for p in [25,50,75]]
    iqr = q3 - q1
    border_min, border_max = q1-1.5*iqr-1e-5, q3+1.5*iqr+1e-5
    if sa_values[0]<=border_min or sa_values[0]>=border_max:
      nearby_samples_bad_sample_key_list.append(sa_key)

  f = open(SAVE_REPORT_PATH+"nearby_comparsion/"+str(target_sample_idx)+"/"+kind+"/report.txt", mode='w')
  f.write(", ".join(nearby_samples_bad_sample_key_list))
  f.close()

  return nearby_comparsion_df, nearby_samples_bad_sample_key_list

# def show_sa_comparsion(sa_df, target_sample_idx, compared_sample_idx_list):
#   for sa_key, sa_series in sa_df.iteritems():
#     target_value = sa_series.iloc[target_sample_idx]
#     compared_value_list = sa_series.iloc[compared_sample_idx_list].values.tolist()
#     Print("   ",sa_key,":",target_value,"|",compared_value_list)

def make_sa_bias_report(sa_df, bias_thresold=0.5):
  high_bias_relative_freequency_list = []

  for sa_key, sa_series in sa_df.iteritems():
    fig = plt.figure(figsize=(5,5))
    plt.title(sa_key)
    plt.hist(sa_series.values, bins=20)
    save_plt_fig(SAVE_REPORT_PATH+"sa_histgram/", sa_key+".png")

    sa_value_relative_freequency = sa_series.value_counts(bins=20).values*1.0/len(sa_series)
    if max(sa_value_relative_freequency>bias_thresold):
      high_bias_relative_freequency_list.append(sa_key)
    # print(sa_key)
    # print(sa_value_relative_freequency)

  f = open(SAVE_REPORT_PATH+"sa_histgram/report.txt", mode='w')
  f.write("High bias (relative freequecy > "+str(bias_thresold)+") skill parameter and state (Check whether the skill parameter is useful):")
  f.write(" "+", ".join(high_bias_relative_freequency_list))
  f.close()

###The true distance does not always match the scatter plot that extracts only pc1 and pc2.
def show_nearby_sample_info(l, kind, dist_M, target_sample_idx, n_consider=5):
  near_sample_idx_list, near_sample_distance_list = make_nearby_sample_list_from_target_sample(dist_M, target_sample_idx)
  Print("  nearby samples info ("+kind+") :")
  for idx, dist in zip(near_sample_idx_list[1:n_consider+1], near_sample_distance_list[1:n_consider+1]):
    Print("    distance:",dist)
    Print("      ","episode",":",idx)
    for key in l.state_histories.keys():
      Print("      ",key,":", l.state_histories[key][idx])
  ## Print("  sa comparsion to nearby samples:")
  ## show_sa_comparsion(sa_df, target_sample_idx, near_sample_idx_list)

def make_nearby_sa_report(l, bad_sample_idx_list, neighbor_threshold=1.0, n_print=5, n_table_rows=10, do_print_nearby_sample_info=False):
  pca_feature_df = make_pca_sa_df(l.input_sa_df, do_plot_pca_info=False)
  bad_samples_pca_feature_df = pca_feature_df.iloc[bad_sample_idx_list,:]
  feature_dist_M = distance.cdist(pca_feature_df, pca_feature_df, metric="euclidean")
  # init_state_dist_M = distance.cdist(l.init_state_df, l.init_state_df, metric="euclidean")

  doubtful_episode_list = [""]

  for idx in bad_sample_idx_list:
    nearby_feature_df, nearby_feature_bad_sample_key_list = make_nearby_comparsion_df(l, "pca_feature", feature_dist_M, idx, neighbor_threshold)
    nearby_init_state_df, nearby_init_state_bad_sample_key_list = make_nearby_comparsion_df(l, "init_state", l.init_state_dist_M, idx, neighbor_threshold)
    if (len(nearby_feature_bad_sample_key_list and l.skill_param_df.keys())>=0) and (nearby_feature_df["distance"].iloc[1]<=neighbor_threshold):
      doubtful_episode_list.append(str(idx))

    if do_print_nearby_sample_info:
      Print("episode:",idx)
      for key in l.state_histories.keys():
        Print(" ",key,":",l.state_histories[key][idx])
      show_nearby_sample_info(l, "pca_feature", feature_dist_M, idx, n_consider=n_print)
      print(nearby_feature_df.iloc[:n_print+1, :])
      show_nearby_sample_info(l, "init_state", l.init_state_dist_M, idx, n_consider=n_print)
      print(nearby_init_state_df.iloc[:n_print+1, :])
      print("")

    for kind, nearby_df in zip(["pca_feature", "init_state"], [nearby_feature_df, nearby_init_state_df]):
      fig, ax = plt.subplots(1,1,figsize=(20,10))
      ax.axis('tight')
      ax.axis('off')
      table = ax.table(
        cellText=nearby_df.values.round(4)[:n_table_rows+1],
        colLabels=nearby_df.columns,
        loc='center',
        colWidths=[.08]*len(nearby_df.columns)
        # bbox=[0,0,1,1]
      )
      table.scale(1, 4)
      table.auto_set_font_size(False)
      table.set_fontsize(10)
      save_plt_fig(SAVE_REPORT_PATH+"nearby_comparsion/"+str(idx)+"/"+kind+"/","table")

  f = open(SAVE_REPORT_PATH+"nearby_comparsion/report.txt", mode='w')
  f.write("Very close to nearby sample's skill parameter (Check skill behavior or true dynamics):")
  f.write(" "+", ".join(doubtful_episode_list))
  f.close()

  bad_samples_anotation_list = [str(ep) for ep in bad_sample_idx_list]
  plt.figure(figsize=(10,10))
  plt.grid()
  plt.scatter(x=pca_feature_df.iloc[:,0], y=pca_feature_df.iloc[:,1], s=8, c="blue")
  plt.scatter(x=bad_samples_pca_feature_df.iloc[:,0], y=bad_samples_pca_feature_df.iloc[:,1], s=8, c="red")
  for i,label in enumerate(bad_samples_anotation_list):
    plt.annotate(label, (bad_samples_pca_feature_df.iloc[i,0], bad_samples_pca_feature_df.iloc[i,1]))
  plt.xlim(min(min(pca_feature_df.iloc[:,0]),min(pca_feature_df.iloc[:,1]))-0.5, max(max(pca_feature_df.iloc[:,0]),max(pca_feature_df.iloc[:,1]))+0.5)
  plt.ylim(min(min(pca_feature_df.iloc[:,0]),min(pca_feature_df.iloc[:,1]))-0.5, max(max(pca_feature_df.iloc[:,0]),max(pca_feature_df.iloc[:,1]))+0.5)
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  # plt.show()
  save_plt_fig(SAVE_REPORT_PATH+"pca_reuslt/","2d_scatter")


def analyze_dynamics(l, targe_state, bad_sample_idx, log_name, neighbor_threshold=0.05, level_of_significance=0.05, power=0.8, n_test_sample=100):
  dynamics_db = load_db(ROOT_PATH,log_name)
  dynamics_state_histories = get_state_histories(dynamics_db, l.all_sa_node_unit_pair)
  skill_param_df = make_sa_df(dynamics_state_histories, l.skill_params_node_unit_pair)
  dynamics_target_state_output_history = dynamics_state_histories[targe_state]
  dynamics_return_history = dynamics_state_histories[".r"]
  # print(skill_param_df)
  bad_samples_target_state_value = l.all_sa_df[targe_state][bad_sample_idx]
  bad_samples_init_state_dict = l.init_state_df.iloc[bad_sample_idx]
  # print(dynamics_target_state_output_history)
  # print(dynamics_return_history)

  ### compare to bad_sample sample
  q1,q2,q3 = [np.percentile(dynamics_target_state_output_history, [p]) for p in [25,50,75]]
  iqr = q3 - q1
  border_min, border_max = q1-2.0*iqr-1e-5, q3+2.0*iqr+1e-5
  v_min, v_max = min(dynamics_target_state_output_history), max(dynamics_target_state_output_history)
  if bad_samples_target_state_value>v_max or bad_samples_target_state_value<v_min:
  # if bad_samples_target_state_value>=border_max or bad_samples_target_state_value<=border_min:
    print("Bad_sample value ("+str(bad_samples_target_state_value)+") is not included in surrounding dynamics observed value range ("+str(v_min)+" ~ "+str(v_max)+").")
    print("Should search more close, or analyze manually")

  ### compare to near initial state
  def judge_whether_near_to_sample(df):
    filter_list = []
    for key in bad_samples_init_state_dict.keys():
      bad_samples_init_state = bad_samples_init_state_dict[key]
      value_range = l.value_range[key]
      if len(value_range)==1:
        range = neighbor_threshold*value_range[0]["range"]
        filter = (((bad_samples_init_state - range) < df[key]) & (df[key] < (bad_samples_init_state + range))).values
        filter_list.append(filter)
      else:
        raise(Exception)
    merged_filter_list = [all(filters) for filters in zip(*filter_list)]
    merged_filter_list[bad_sample_idx] = False
    return merged_filter_list

  nearby_init_state_df = l.all_sa_df[judge_whether_near_to_sample(l.all_sa_df)]
  nearby_init_state_better_return_df = nearby_init_state_df.sort_values(".r", ascending=False).head(int(len(nearby_init_state_df)/2.))
  print(nearby_init_state_better_return_df)
  nearby_init_state_better_return_mearn = nearby_init_state_better_return_df[".r"].mean()
  dynamics_return_mean = np.mean(dynamics_return_history)

  eval_dynamics = "good" if dynamics_return_mean >= -0.5 else "bad"
  eval_nearby_init_state = "good" if nearby_init_state_better_return_mearn >= -0.5 else "bad"
  print("Average of surrounding dynamics return is "+str(eval_dynamics)+" ("+str(dynamics_return_mean)+").")
  print("Average of return log which is nearby initstate and included in higher 50% is "+str(eval_nearby_init_state)+" ("+str(nearby_init_state_better_return_mearn)+").")
  if eval_nearby_init_state=="bad":
    bad_init_state = l.init_state_df.iloc[bad_sample_idx]
    print("Danger init state value range ...")
    for sa_key in bad_init_state.index:
      value_range = l.value_range[sa_key]
      if len(value_range)==1:
        denger_range_min = max(bad_init_state[sa_key] - 0.05*value_range[0]["range"], value_range[0]["min"])
        denger_range_max = min(bad_init_state[sa_key] + 0.05*value_range[0]["range"], value_range[0]["max"])
        Print(sa_key,":",str(denger_range_min),"~",str(denger_range_max))
      else:
        raise(Exception)

  ### Lasso
  for column in skill_param_df.columns:
    skill_param_df[column+"**2"] = skill_param_df[column]**2

  scaler = MinMaxScaler()
  regressor = Lasso(alpha=1e-2)
  scaler.fit(skill_param_df)
  regressor.fit(scaler.transform(skill_param_df), dynamics_target_state_output_history)

  n_dummy = 10
  counter_list = [0]*len(skill_param_df.columns)
  #if p0=0.5, p=0.9, alpha=0.05, beta=0.8, n_test_sample shold be more than 7.
  for _ in range(n_test_sample):
    dummy_df = pd.DataFrame()
    for i in range(n_dummy):
      dummy_df["dummy"+str(i+1)] = np.random.rand(len(skill_param_df))

    scaler = MinMaxScaler()
    regressor = Lasso(alpha=1e-2)
    merged_df = skill_param_df.join(dummy_df)
    scaler.fit(merged_df)
    regressor.fit(scaler.transform(merged_df), dynamics_target_state_output_history)
    max_dummy_coef = max(abs(regressor.coef_[-n_dummy:]))
    # print(regressor.coef_)
    # print(max_dummy_coef)
    for idx in np.where(abs(np.array(regressor.coef_)[:-n_dummy]) > max_dummy_coef)[0]:
      # print(idx)
      counter_list[idx] += 1
      # print(counter_list)
  print(counter_list)
  effective_skill_params = []
  for count, skill in zip(counter_list, skill_param_df.columns):
    p = binom_test(count, n_test_sample, 0.5)
    # print(skill, p, p < level_of_significance, count*1.0/n_test_sample, count/n_test_sample > 0.5)
    if (p < level_of_significance) & (count*1.0/n_test_sample > 0.5):
      if "**2" in skill:
        skill = skill[:-3]
      effective_skill_params.append(skill)
  effective_skill_params = list(set(effective_skill_params))
  Print("Effective skill params :", effective_skill_params)

  if len(effective_skill_params)>=1:
    for skill in effective_skill_params:
      bad_sample_skill_param_section = l.all_sa_df[skill].value_counts(bins=20).index[0]
      bad_sample_skill_param_value = l.all_sa_df[skill][bad_sample_idx]
      is_in_peak = bad_sample_skill_param_value in bad_sample_skill_param_section
      is_limit_value = max(l.all_sa_df[skill]) in bad_sample_skill_param_section or min(l.all_sa_df[skill]) in bad_sample_skill_param_section
      if is_in_peak and is_limit_value:
        Print(skill,"is limit value on bad sample ("+str(bad_sample_skill_param_value)+").")

def make_bad_sample_info(ct, l, target_state, log_name):
  global SAVE_REPORT_PATH
  SAVE_REPORT_PATH = SAVE_REPORT_ROOT_PATH + "/" + log_name + "/" +target_state + "/"

  history = l.state_histories[target_state]
  bad_sample_idx_list = l.bad_sample_idx_dict[target_state]

  make_sa_bias_report(l.all_sa_df)
  make_nearby_sa_report(l, bad_sample_idx_list, neighbor_threshold=0.5)

  color_list = ["red" if idx in bad_sample_idx_list else "blue" for idx in range(len(history))]
  ct.Run('mysim.vis.out_true_est2', log_name, target_state, color_list)
  save_plt_fig(SAVE_REPORT_PATH,"learning_curve")


def Run(ct, *args):
  log_name = args[0]
  l = TContainer(debug=True)

  l.init_states_node_unit_pair = [
    ["size_srcmouth", ["n0"], 1],
    # ["material2", ["n0"], 1],
  ]
  l.target_states_node_unit_pair = [
    ["da_pour", ["n4tir","n4sar"], 1/0.0055],
    ["da_spill2", ["n4tir","n4sar"], 10],
  ]
  l.another_state_node_unit_pair= [
    [".r", ["n4tir", "n4sar"], 1],
    # ["p_pour", ["n2b"], 1],
    # ["flow_var", ["n4tir","n4sar"], 1],
    # ["lp_flow", ["n4tir","n4sar"], 1],
  ]
  l.skill_params_node_unit_pair = [
    ["p_pour_trg", ["n0"], 1],
    # ["dtheta1", ["n0"], 1],
    # ["dtheta2", ["n4tir"], 1],
    ["shake_spd", ["n0"], 1],
    ["shake_axis2", ["n0"], 1],
    # ["skill", ["n0"], 1]
  ]
  l.value_range = {
    "size_srcmouth": [{"min": 0.03, "max": 0.08, "range": 0.05}],
    "p_pour_trg": [{"min": 0.2, "max": 1.2, "range": 1.0}, {"min": 0.1, "max": 0.7, "range": 0.6}],
    "dtheta2": [{"min": 0.002, "max": 0.02, "range": 0.018}],
    "shake_spd": [{"min": 0.2, "max": 0.4, "range": 0.2}],
    "shake_axis2": [{"min": 0.01, "max": 0.03, "range": 0.02}, {"min": -0.2*math.pi, "max": 0.2*math.pi, "range": 0.4*math.pi}]
  }
  l.target_value_dict = {
    "da_pour": 0.3,
    "da_spill2": 0,
  }
  l.reward_function = lambda df, idx: - 100*max(0, l.targe_value_dict["da_pour"][idx] - df["da_pour"][idx])**2 - max(0, df["da_pour"][idx] - l.targe_value_dict["da_pour"][idx])**2 - max(0, df["da_spill2"][idx])**2

  l.input_sa_node_unit_pair = l.init_states_node_unit_pair + l.skill_params_node_unit_pair
  l.all_sa_node_unit_pair = l.input_sa_node_unit_pair + l.target_states_node_unit_pair + l.another_state_node_unit_pair
  l.db = load_db(ROOT_PATH,log_name)
  l.state_histories = get_state_histories(l.db, l.all_sa_node_unit_pair)
  l.input_sa_df = make_sa_df(l.state_histories, l.input_sa_node_unit_pair)
  l.all_sa_df = make_sa_df(l.state_histories, l.all_sa_node_unit_pair)
  l.init_state_df = make_sa_df(l.state_histories, l.init_states_node_unit_pair)
  l.skill_param_df = make_sa_df(l.state_histories, l.skill_params_node_unit_pair)
  l.init_state_dist_M = distance.cdist(l.init_state_df, l.init_state_df, metric="euclidean")

  WINDOW_SIZE = 50

  # exit_state_dict = defaultdict(list)
  # diff_threshold = 1e-3
  # move_diff_to_trg_list = np.sqrt((all_sa_df["p_pour_trg_dim1"] - all_sa_df["p_pour_dim1"])**2 + (all_sa_df["p_pour_trg_dim2"] - all_sa_df["p_pour_dim3"])**2).tolist()
  # for move_diff_to_trg in move_diff_to_trg_list:
  #   if move_diff_to_trg > diff_threshold:
  #     exit_state_dict["p_pour"].append("collision")
  #   else:
  #     exit_state_dict["p_pour"].append("reached")

  # fig = plt.figure()
  # plt.hist(move_diff_to_trg_list, bins=30)
  # plt.show()
  # fig = plt.figure()
  # plt.hist(exit_state_dict["lp_pour"], bins=2)
  # plt.show()

  # target_state = "da_spill2"
  all_bad_sample_idx_list = []
  l.bad_sample_idx_dict = defaultdict(list)
  for target_state in np.array(l.target_states_node_unit_pair)[:,0]:
    global SAVE_REPORT_PATH
    SAVE_REPORT_PATH = SAVE_REPORT_ROOT_PATH + "/" +target_state + "/"
    l.bad_sample_idx_dict[target_state] = make_bad_sample_idx_list(target_state, l.state_histories, WINDOW_SIZE=WINDOW_SIZE, IQR_MAGNIFICATION=2.0, return_threshold=-1.0)
    all_bad_sample_idx_list += l.bad_sample_idx_dict[target_state]
  all_bad_sample_idx_list = np.sort(list(set(all_bad_sample_idx_list)))
    
  n_sample = 30
  searach_range = 0.05
  bad_sample_simulation_input_sa_dict_list = []
  # for idx in bad_sample_idx_list[:1]:
  for idx in all_bad_sample_idx_list:
    bad_sample_simulation_input_sa_dict = defaultdict(list)
    for sa_key,_,_ in l.input_sa_node_unit_pair:
      v = l.state_histories[sa_key][idx]
      is_scalar = False
      if type(v)!=np.ndarray:
        v = [v]
        is_scalar = True
      for _ in range(n_sample):
        if sa_key in np.array(l.skill_params_node_unit_pair)[:,0]:
          v_noise = deepcopy(v)
          for i in range(len(v_noise)):
            v_noise_i_min = max(v[i] - searach_range/2*l.value_range[sa_key][i]["range"], l.value_range[sa_key][i]["min"])
            v_noise_i_max = min(v[i] + searach_range/2*l.value_range[sa_key][i]["range"], l.value_range[sa_key][i]["max"])
            v_noise[i] = (v_noise_i_max - v_noise_i_min) * np.random.rand() + v_noise_i_min
          if is_scalar==True: v_noise = v_noise[0]
          bad_sample_simulation_input_sa_dict[sa_key].append(v_noise)
        elif sa_key in np.array(l.init_states_node_unit_pair)[:,0]:
          if is_scalar==True:
            bad_sample_simulation_input_sa_dict[sa_key].append(v[0])
          else:
            bad_sample_simulation_input_sa_dict[sa_key].append(v)
    bad_sample_simulation_input_sa_dict_list.append(bad_sample_simulation_input_sa_dict)

  # for idx, bad_sample_simulation_input_sa_dict in zip(all_bad_sample_idx_list, bad_sample_simulation_input_sa_dict_list):
  #   ct.Run("mysim.debugger.experiments.ex5", "sampling", bad_sample_simulation_input_sa_dict, ROOT_PATH+log_name+"/sampling/"+str(idx)+"/") 

  for target_state in np.array(l.target_states_node_unit_pair)[:,0]:
    print("")
    Print("target_state :",target_state)

    make_bad_sample_info(ct, l, target_state, log_name)

    for idx in l.bad_sample_idx_dict[target_state]:
      Print("episode :",idx)
      analyze_dynamics(l, target_state, idx, log_name+"/sampling/"+str(idx)+"/")
      print("")
  print("")
  # analyze_dynamics(l, target_state, bad_sample_idx_list[0], log_name+"/sampling/"+str(bad_sample_idx_list[0])+"/")

