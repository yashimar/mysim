from core_tool import *
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib
from scipy.spatial import distance
from scipy.stats import norm
from collections import defaultdict

def Help():  
  return '''Visualize dpl logs.
  Usage: mysim.vis_log'''

def Plot(df, ylabel,label, xmin=0, ymin=-30):
  for i in range(len(df.columns)):
    plt.plot(df.iloc[:,i], label=label)
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    plt.xlim(xmin,len(df))
    plt.ylim(ymin,0)
    plt.legend()
    plt.grid()
  plt.show()

def MtrScatter(df, mtr_list, vis_mtr_list, xmin=0, ymin=-30, ymax=0, s=10, marker="o", label=""):
  c_dict = {"bounce":"purple","nobounce":"green","natto":"orange","ketchup":"red"}
  for mtr in list(set(mtr_list)):
    if mtr in vis_mtr_list:
      mtr_ids = [i for i, x in enumerate(mtr_list) if x==mtr]
      plt.scatter(mtr_ids, df.iloc[mtr_ids], label=mtr+" "+label, c=c_dict[mtr], s=s, marker=marker)
  plt.xlim(xmin,len(mtr_list))
  plt.ylim(ymin,ymax)
  plt.legend()
  plt.show()

def PlotMeanStd(df, df_error, ylabel):
  plt.errorbar(df.index, df, df_error)
  plt.xlabel("episode")
  plt.ylabel(ylabel)
  plt.ylim(-30,0)
  plt.legend(loc='lower right')
  plt.grid()
  plt.show()

def mahalanobis_plot(args, name, ymin, ymax, figsize, window,true_list, est_mean_list, est_var_list):
  d_list = [distance.mahalanobis(true, est_mena, 1/est_var) for (true, est_mena, est_var) in zip(true_list, est_mean_list, est_var_list)]
  d_list_ma = pd.Series(d_list).rolling(window).mean()
  fig = plt.figure(figsize=figsize)
  plt.title(args[0]+" mahalabobis "+name)
  plt.ylim(ymin,ymax)
  plt.plot(d_list, label="mahalanobis distance")
  plt.plot(d_list_ma, label="mahalanobis distance (ma window: "+str(window)+")", c="orange")
  plt.xlabel("episode")
  plt.legend()
  plt.show()

def probability_plot(args, name, ymin, ymax, figsize, window,true_list, est_mean_list, est_var_list):
  def calc_p(x, mu, var):
    z = (x-mu)/np.sqrt(var)
    p = 2*norm.sf(z) if z>=0 else 2*norm.cdf(z)
    return p
  p_list = [calc_p(true, est_mean, est_var) for (true, est_mean, est_var) in zip(true_list, est_mean_list, est_var_list)]
  p_list_ma = pd.Series(p_list).rolling(window).mean()
  fig = plt.figure(figsize=figsize)
  plt.title(args[0]+" bothside probability "+name)
  plt.ylim(ymin,ymax)
  plt.plot(p_list, label="bothside probability")
  plt.plot(p_list_ma, label="bothside probability (ma window: "+str(window)+")", c="orange")
  plt.xlabel("episode")
  plt.legend()
  plt.show()

def mean_var_plot(args, ymin, ymax, border, figsize, name, true_list, est_mean_list, est_var_list, low_r_list, mtr_list, vis_mtr_list):
  fig = plt.figure(figsize=figsize)
  plt.title(args[0]+" E["+name+"]") 
  plt.ylim(ymin,ymax)
  plt.errorbar(
    np.arange(len(est_mean_list)),est_mean_list,2*np.sqrt(est_var_list),
    ecolor="pink",
    label="E["+name+"] +/- 2*sqrt( Var["+name+"] )")
  low_r_list = [ep for ep in low_r_list if ep > 50]
  plt.vlines(low_r_list,ymin,ymax,
              linestyles="dashed",colors="gray",
              label="low r episode",alpha=0.3)
  # plt.plot([border]*len(est_mean_list), linestyle="dashed", alpha=0.5, c="black", label="border ("+str(border)+")")
  # MtrScatter(pd.DataFrame(mean_spill), mtr_list, vis_mtr_list, ymin=ymin, ymax=ymax, s=20, marker="o", label="est mean")
  # MtrScatter(pd.DataFrame(true_list), mtr_list, vis_mtr_list, ymin=ymin, ymax=ymax, s=20, marker="*", label="true")
  plt.scatter(np.arange(len(true_list)), true_list, c="black", marker="*", label="true")
  plt.legend()
  plt.subplots_adjust(left=0.05, right=0.95)
  plt.show()  

def Run(ct, *args):
  name_log = args[0]
  r_thr = -1.0
  
  data_list = []
  mtr_list = []
  smsz_list = []
  low_r_list = []
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  log_path = root_path + name_log + "/dpl_est.dat"
  sl_path = root_path + name_log + "/sequence_list.yaml"
  # db_path = root_path + name_log + "/database.yaml"
  config_path = root_path + name_log + "/config_log.yaml"
  # data_list.append(np.genfromtxt(log_path))

  d = defaultdict(list)
  true_spill = []
  true_pour = []
  true_da_total = []
  true_lp_flow_x = []
  true_lp_flow_y = []
  true_flow_var = []
  true_lp_pour_x = []
  true_lp_pour_z = []
  with open(sl_path, "r") as yml:
    sl = yaml.load(yml)
  # with open(db_path, "r") as yml:
  #   db = yaml.load(yml)["Entry"]
  for ep in range(len(sl)):
    try:
      true_spill.append(sl[ep]["reward"][3][2][0]["da_spill2"])
      true_pour.append(sl[ep]["reward"][3][3][0]["da_pour"])
      true_da_total.append(sl[ep]["sequence"][5]["n4sar"]["da_total"][0][0])
      true_lp_flow_x.append(sl[ep]["sequence"][5]["n4sar"]["lp_flow"][0][0])
      true_lp_flow_y.append(sl[ep]["sequence"][5]["n4sar"]["lp_flow"][1][0])
      true_flow_var.append(sl[ep]["sequence"][5]["n4sar"]["flow_var"][0][0])
      true_lp_pour_x.append(sl[ep]["sequence"][3]["n2b"]["lp_pour"]["X"][0][0])
      true_lp_pour_z.append(sl[ep]["sequence"][3]["n2b"]["lp_pour"]["X"][2][0])

      # true_da_total.append(db[ep]["Seq"][8]["XS"]["da_total"]["X"][0][0])
      # true_lp_flow_x.append(db[ep]["Seq"][8]["XS"]["lp_flow"]["X"][0][0])
      # true_lp_flow_y.append(db[ep]["Seq"][8]["XS"]["lp_flow"]["X"][1][0])
      # true_flow_var.append(db[ep]["Seq"][8]["XS"]["flow_var"]["X"][0][0])
      # true_lp_pour_x.append(db[ep]["Seq"][7]["XS"]["lp_pour"]["X"][0][0])
      # true_lp_pour_z.append(db[ep]["Seq"][7]["XS"]["lp_pour"]["X"][2][0])

      # d["gh_abs"].append(db[ep]["Seq"][8]["XS"]["gh_abs"]["X"][0][0])
      # d["da_trg"].append(db[ep]["Seq"][8]["XS"]["da_trg"]["X"][0][0])
      # d["dtheta1"].append(db[ep]["Seq"][8]["XS"]["dtheta1"]["X"][0][0])
      # d["shake_spd"].append(db[ep]["Seq"][8]["XS"]["shake_spd"]["X"][0][0])
      # d["shake_axis2_range"].append(db[ep]["Seq"][8]["XS"]["shake_axis2"]["X"][0][0])
      # d["shake_axis2_angle"].append(db[ep]["Seq"][8]["XS"]["shake_axis2"]["X"][1][0])
    except:
      pass

  # gh_abs = map( lambda ep: ep["Seq"][8]["XS"]["gh_abs"]["X"][0][0], db )
  # da_trg = map( lambda ep: ep["Seq"][8]["XS"]["da_trg"]["X"][0][0], db )
  # # smsz = map( lambda ep: ep["Seq"][8]["XS"]["size_srcmouth"]["X"][0][0], db )
  # # material2 = 
  # dtheta1 = map( lambda ep: ep["Seq"][8]["XS"]["dtheta1"]["X"][0][0], db )
  # shake_spd = map( lambda ep: ep["Seq"][8]["XS"]["shake_spd"]["X"][0][0], db )
  # shake_axis2_range = map( lambda ep: ep["Seq"][8]["XS"]["shake_axis2"]["X"][0][0], db )
  # shake_axis2_angle = map( lambda ep: ep["Seq"][8]["XS"]["shake_axis2"]["X"][1][0], db )


  data_list = []
  with open(log_path, "r") as log_data:
    for ep, line in enumerate(log_data):
      line = line.split("\n")[0].split(" ")
      line = map(lambda x: float(x), line)
      data_list.append(line)
      if line[1] < r_thr:
        low_r_list.append(ep)
  with open(config_path, "r") as yml:
    config = yaml.load(yml)
  for i in range(len(config)):
    conf = config[i]
    if conf["ContactBounce"]==0.7: mtr_list.append("bounce")
    elif conf["ViscosityParam1"]==1.5e-06: mtr_list.append("natto")
    elif conf["ViscosityParam1"]==2.5e-07: mtr_list.append("ketchup")
    else: mtr_list.append("nobounce")
    smsz_list.append(conf["SrcSize2H"])
  learn_ep_list = map(lambda x: int(x[0]), data_list)
  mtr_list = [mtr for j,mtr in enumerate(mtr_list) if j in learn_ep_list]
  smsz_list = [smsz for j,smsz in enumerate(smsz_list) if j in learn_ep_list]
  learn_ep_before_sampling_list = [i for i in range(len(learn_ep_list)-1) 
                                  if learn_ep_list[i+1]-learn_ep_list[i]>=2]

  window = 10
  df_list = []
  df_est_n0_list = []
  df_est_last_list = []
  df_ma_list = []
  for i, data in enumerate(data_list):
    df_list.append(data[1])
    df_est_n0_list.append(data[2])
    df_est_last_list.append(data[-1])
  df = pd.DataFrame(df_list)
  df_est_n0 = pd.DataFrame(df_est_n0_list)
  df_est_last = pd.DataFrame(df_est_last_list)

  mean_pour = []
  var_pour = []
  sq_mean_pour = []
  sq_var_pour = []
  mean_spill = []
  var_spill = []
  sq_mean_spill = []
  sq_var_spill = []
  mean_da_total = []
  var_da_total = []
  mean_lp_flow_x = []
  var_lp_flow_x = []
  mean_lp_flow_y = []
  var_lp_flow_y = []
  mean_flow_var = []
  var_flow_var = []
  mean_lp_pour_x = []
  var_lp_pour_x = []
  mean_lp_pour_z = []
  var_lp_pour_z = []
  for i in range(len(df)):
    with open(root_path+name_log+"/best_est_trees/ep"+str(i)+"_n0.jb", mode="rb") as f:
      tree = joblib.load(f)
      mp = tree.Tree[TPair("n4sar",0)].XS["da_pour"].X.item()
      vp = tree.Tree[TPair("n4sar",0)].XS["da_pour"].Cov.item()
      ms = tree.Tree[TPair("n4sar",0)].XS["da_spill2"].X.item()
      vs = tree.Tree[TPair("n4sar",0)].XS["da_spill2"].Cov.item()
      # mp = mp - 0.3
      mp2 = mp**2 + vp
      vp2 = 2*vp**2 + 4*mp**2*vp
      ms2 = ms**2 + vs
      vs2 = 2*vs**2 + 4*ms**2*vs
      mean_pour.append(mp)
      var_pour.append(vp)
      mean_spill.append(ms)
      var_spill.append(vs)
      sq_mean_spill.append(ms2)
      sq_var_spill.append(vs2)
      mean_da_total.append(tree.Tree[TPair("n4sar",0)].XS["da_total"].X.item())
      var_da_total.append(tree.Tree[TPair("n4sar",0)].XS["da_total"].Cov.item())
      mean_lp_flow_x.append(tree.Tree[TPair("n4sar",0)].XS["lp_flow"].X.tolist()[0][0])
      var_lp_flow_x.append(tree.Tree[TPair("n4sar",0)].XS["lp_flow"].Cov.tolist()[0][0])
      mean_lp_flow_y.append(tree.Tree[TPair("n4sar",0)].XS["lp_flow"].X.tolist()[1][0])
      var_lp_flow_y.append(tree.Tree[TPair("n4sar",0)].XS["lp_flow"].Cov.tolist()[1][1])
      mean_flow_var.append(tree.Tree[TPair("n4sar",0)].XS["flow_var"].X.item())
      var_flow_var.append(tree.Tree[TPair("n4sar",0)].XS["flow_var"].Cov.item())
      mean_lp_pour_x.append(tree.Tree[TPair("n4sar",0)].XS["lp_pour"].X.tolist()[0][0])
      var_lp_pour_x.append(tree.Tree[TPair("n4sar",0)].XS["lp_pour"].Cov.tolist()[0][0])
      mean_lp_pour_z.append(tree.Tree[TPair("n4sar",0)].XS["lp_pour"].X.tolist()[2][0])
      var_lp_pour_z.append(tree.Tree[TPair("n4sar",0)].XS["lp_pour"].Cov.tolist()[2][2])
  # print(tree.Tree[TPair("n4sar",0)].XS)
  # mean_pour = pd.DataFrame(mean_pour)
  # var_pour = pd.DataFrame(var_pour)
  # mean_spill = pd.DataFrame(mean_spill)
  # var_spill = pd.DataFrame(var_spill)
  # sq_mean_spill = pd.DataFrame(sq_mean_spill)
  # sq_var_spill = pd.DataFrame(sq_var_spill)

  ### plot
  vis_mtr_list = [
    "bounce", 
    "nobounce", 
    "natto", 
    "ketchup"
  ]
  if False:
    fig = plt.figure(figsize=(20,5))
    plt.title(args[0]) 
    xmin = 0
    ymin = -10
    border = -1.0
    plt.plot([border]*len(df), linestyle="dashed", alpha=0.5, c="red", label="border ("+str(border)+")")

    Plot(df.iloc[:,:], "return", "return", xmin, ymin)
    MtrScatter(df.iloc[:,:], mtr_list, vis_mtr_list, xmin, ymin)
    
    Plot(df_est_n0.iloc[:,:], "return", "estimate_n0", xmin , ymin)
    # MtrScatter(df_est_n0.iloc[:,:], mtr_list, vis_mtr_list, xmin, ymin)

    # Plot(df_est_last.iloc[:,:], "return", "estimate_last", xmin , ymin)
    # MtrScatter(df_est_last.iloc[:,:], mtr_list, vis_mtr_list, xmin, ymin)

    low_r_list = [ep for ep in low_r_list if ep > 50]
    plt.vlines(low_r_list,ymin,0,
                linestyles="dashed",colors="gray",
                label="low r episode")
    
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()

  if False:
    ymin, ymax = 0., 0.005
    fig = plt.figure(figsize=(20,5))
    plt.title(args[0]+" Cov[da_pour]") 
    plt.plot(var_pour, label="Cov[da_pour]")
    MtrScatter(pd.DataFrame(var_pour), mtr_list, vis_mtr_list, ymin=ymin, ymax=ymax, s=20)
    low_r_list = [ep for ep in low_r_list if ep > 50]
    plt.vlines(low_r_list,ymin,ymax,
                linestyles="dashed",colors="gray",
                label="low r episode")
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()

  if False:
    ymin, ymax = 0., 0.1
    fig = plt.figure(figsize=(20,5))
    plt.title(args[0]+" Cov[da_spill2]") 
    plt.plot(var_spill, label="Cov[da_spill2]")
    MtrScatter(pd.DataFrame(var_spill), mtr_list, vis_mtr_list, ymin=ymin, ymax=ymax, s=20)
    low_r_list = [ep for ep in low_r_list if ep > 50]
    plt.vlines(low_r_list,ymin,ymax,
                linestyles="dashed",colors="gray",
                label="low r episode")
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()

  if False:
    mean_var_plot(
      ymin=0, ymax=5.0, border=0.5, figsize=(20,5),
      name="da_spill2", args=args, 
      true_list=true_spill, est_mean_list=mean_spill, est_var_list=var_spill, 
      low_r_list=low_r_list, mtr_list=mtr_list, vis_mtr_list=vis_mtr_list
    )
  if False:
    mean_var_plot(
      ymin=0.3, ymax=0.6, border=0.3, figsize=(20,5),
      name="da_pour", args=args, 
      true_list=true_pour, est_mean_list=mean_pour, est_var_list=var_pour, 
      low_r_list=low_r_list, mtr_list=mtr_list, vis_mtr_list=vis_mtr_list
    )
  if False:
    mean_var_plot(
      ymin=0.4, ymax=0.6, border=0.3, figsize=(20,5),
      name="da_total", args=args, 
      true_list=true_da_total, est_mean_list=mean_da_total, est_var_list=var_da_total, 
      low_r_list=low_r_list, mtr_list=mtr_list, vis_mtr_list=vis_mtr_list
    )
  if True:
    mean_var_plot(
      ymin=-0.4, ymax=0.4, border=0.3, figsize=(20,5),
      name="lp_flow_x", args=args, 
      true_list=true_lp_flow_x, est_mean_list=mean_lp_flow_x, est_var_list=var_lp_flow_x, 
      low_r_list=low_r_list, mtr_list=mtr_list, vis_mtr_list=vis_mtr_list
    )
  if False:
    mean_var_plot(
      ymin=-0.2, ymax=0.2, border=0.3, figsize=(20,5),
      name="lp_flow_y", args=args, 
      true_list=true_lp_flow_y, est_mean_list=mean_lp_flow_y, est_var_list=var_lp_flow_y, 
      low_r_list=low_r_list, mtr_list=mtr_list, vis_mtr_list=vis_mtr_list
    )
  if False:
    mean_var_plot(
      ymin=0, ymax=1.0, border=0.3, figsize=(20,5),
      name="flow_var", args=args, 
      true_list=true_flow_var, est_mean_list=mean_flow_var, est_var_list=var_flow_var, 
      low_r_list=low_r_list, mtr_list=mtr_list, vis_mtr_list=vis_mtr_list
    )
  if False:
    mean_var_plot(
      ymin=-0.3, ymax=0.1, border=0.3, figsize=(20,5),
      name="lp_pour_x", args=args, 
      true_list=true_lp_pour_x, est_mean_list=mean_lp_pour_x, est_var_list=var_lp_pour_x, 
      low_r_list=low_r_list, mtr_list=mtr_list, vis_mtr_list=vis_mtr_list
    )
  if False:
    mean_var_plot(
      ymin=0, ymax=0.4, border=0.3, figsize=(20,5),
      name="lp_pour_z", args=args, 
      true_list=true_lp_pour_z, est_mean_list=mean_lp_pour_z, est_var_list=var_lp_pour_z, 
      low_r_list=low_r_list, mtr_list=mtr_list, vis_mtr_list=vis_mtr_list
    )
  

  if False:
    mahalanobis_plot(
      args=args, name="da_pour", 
      ymin=0.0, ymax=8.0, figsize=(20,5), window=10, 
      true_list=true_pour, est_mean_list=mean_pour, est_var_list=var_pour
    )
  if False:
    mahalanobis_plot(
      args=args, name="da_spill2", 
      ymin=0.0, ymax=8.0, figsize=(20,5), window=10, 
      true_list=true_spill, est_mean_list=mean_spill, est_var_list=var_spill
    )
  if False:
    mahalanobis_plot(
      args=args, name="da_total", 
      ymin=0.0, ymax=8.0, figsize=(20,5), window=10, 
      true_list=true_da_total, est_mean_list=mean_da_total, est_var_list=var_da_total
    )
  if False:
    mahalanobis_plot(
      args=args, name="lp_flow_x", 
      ymin=0.0, ymax=8.0, figsize=(20,5), window=10, 
      true_list=true_lp_flow_x, est_mean_list=mean_lp_flow_x, est_var_list=var_lp_flow_x
    )
  if False:
    mahalanobis_plot(
      args=args, name="lp_flow_y", 
      ymin=0.0, ymax=8.0, figsize=(20,5), window=10, 
      true_list=true_lp_flow_y, est_mean_list=mean_lp_flow_y, est_var_list=var_lp_flow_y
    )
  if False:
    mahalanobis_plot(
      args=args, name="flow_var", 
      ymin=0.0, ymax=8.0, figsize=(20,5), window=10, 
      true_list=true_flow_var, est_mean_list=mean_flow_var, est_var_list=var_flow_var
    )
  if False:
    mahalanobis_plot(
      args=args, name="lp_pour_x", 
      ymin=0.0, ymax=6.0, figsize=(20,5), window=10, 
      true_list=true_lp_pour_x, est_mean_list=mean_lp_pour_x, est_var_list=var_lp_pour_x, 
    )
  if False:
    mahalanobis_plot(
      args=args, name="lp_pour_z", 
      ymin=0.0, ymax=6.0, figsize=(20,5), window=10, 
      true_list=true_lp_pour_z, est_mean_list=mean_lp_pour_z, est_var_list=var_lp_pour_z, 
    )

  if False:
    probability_plot(
      args=args, name="da_pour", 
      ymin=0.0, ymax=1.0, figsize=(20,5), window=10, 
      true_list=true_pour, est_mean_list=mean_pour, est_var_list=var_pour
    )
  if False:
    probability_plot(
      args=args, name="da_spill2", 
      ymin=0.0, ymax=1.0, figsize=(20,5), window=10, 
      true_list=true_spill, est_mean_list=mean_spill, est_var_list=var_spill
    )
  if False:
    probability_plot(
      args=args, name="da_total", 
      ymin=0.0, ymax=1.0, figsize=(20,5), window=10, 
      true_list=true_da_total, est_mean_list=mean_da_total, est_var_list=var_da_total
    )
  if False:
    probability_plot(
      args=args, name="lp_flow_x", 
      ymin=0.0, ymax=1.0, figsize=(20,5), window=10, 
      true_list=true_lp_flow_x, est_mean_list=mean_lp_flow_x, est_var_list=var_lp_flow_x
    )
  if False:
    probability_plot(
      args=args, name="lp_flow_y", 
      ymin=0.0, ymax=1.0, figsize=(20,5), window=10, 
      true_list=true_lp_flow_y, est_mean_list=mean_lp_flow_y, est_var_list=var_lp_flow_y
    )
  if False:
    probability_plot(
      args=args, name="flow_var", 
      ymin=0.0, ymax=1.0, figsize=(20,5), window=10, 
      true_list=true_flow_var, est_mean_list=mean_flow_var, est_var_list=var_flow_var
    )
  if False:
    probability_plot(
      args=args, name="lp_pour_x", 
      ymin=0.0, ymax=1.0, figsize=(20,5), window=10, 
      true_list=true_lp_pour_x, est_mean_list=mean_lp_pour_x, est_var_list=var_lp_pour_x, 
    )
  if False:
    probability_plot(
      args=args, name="lp_pour_z", 
      ymin=0.0, ymax=1.0, figsize=(20,5), window=10, 
      true_list=true_lp_pour_z, est_mean_list=mean_lp_pour_z, est_var_list=var_lp_pour_z, 
    )

  # plt.savefig("/home/yashima/Pictures/dpl3_7_21/"+name_log+".png")
  # plt.close()