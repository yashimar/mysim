from core_tool import *
from scipy.stats.stats import mode
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
import joblib
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import pandas as pd

def Help():
  pass

def RwdModel():
  modeldir= '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'\
            +'reward_model'+"/"
  FRwd= TNNRegression()
  prefix= modeldir+'p1_model/FRwd3'
  FRwd.Load(LoadYAML(prefix+'.yaml'), prefix)
  FRwd.Init()

  return FRwd

def Run(ct, *args):
  
  #set model_path, domain, target_model, inputs, output_var_idx

  #set model_path
  root_path = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'
  name_log = args[0]
  model_path = root_path + name_log + "/models/"
  tree_path = root_path+name_log+"/best_est_trees/"

  #set target_model
  # target_model = "Fmvtopour2"
  # target_model = "Fflowc_tip10"
  target_model = "Fflowc_shakeA10"
  # target_model = "Rdamoount"

  #set inputs
  x_values = np.linspace(-0.1,0.0 , 10)    #lp_pour_x
  y_values = np.linspace(0.03, 0.08, 10)[::-1]     #smsz
  # x_values = np.linspace(-0.3, 0.4, 100)    #lp_pour_x
  # y_values = np.linspace(0.1, 0.5, 100)[::-1]     #lp_pour_z
  # x_values = np.linspace(0.35, 0.6, 100)    #p_pour_trg_x
  # y_values = np.linspace(0.1, 0.6, 100)[::-1]     #p_pour_trg_z
  # x_values = np.linspace(0.035, 0.075, 100)
  # y_values = [0]
  inputs = []
  for y in y_values:
    for x in x_values:
      inputs.append([
        # x - 0.6,
        # 0,
        # y - 0.202,
        # x + 0.6,
        # 0.32476305961608887 + 0.202,
        x,
        0,
        0.32476305961608887,
        y,
        # 0.028,
        # -1.22,
        # 0.3,
        # 0.55
        # 0.538 - 0.6,
        # 0,
        # 0.535 - 0.202,
        # x
      ])
  inputs = np.array(inputs)
  
  #set output_var_idx
  output_var_idx = 0

  #set fig params
  fig_title = "leran7 ketchup's r_amount (Taylpr) estimation heatmap"
  subtitle = "lp_pour_trg = 0.325" \
            #  + ", shake_axis2 = (0.01, 0)" \
            #  + "referenced p_pour_trg = (0.43, 0.15) and (0.45, 0.11)"
  fig_xlabel = "lp_pour_trg_x"
  fig_ylabel = "smsz"
  
  #set domain
  domain= TGraphDynDomain()
  SP= TCompSpaceDef
  domain.SpaceDefs={
    'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position (x,z)
    'dtheta2': SP('action',1,min=[0.002],max=[0.005]),  #Pouring skill parameter for 'std_pour'
    'shake_axis2': SP('action',2,min=[0.01,-0.5*math.pi],max=[0.1,0.5*math.pi]),  #Pouring skill parameter for 'shake_A'
    'ps_rcv': SP('state',12),  #4 edge point positions (x,y,z)*4 of receiver
    'lp_pour': SP('state',3),  #Pouring axis position (x,y,z) in receiver frame
    "da_trg": SP("state",1),
    "a_src": SP("state",1),
    'da_pour': SP('state',1),  #Amount poured in receiver (displacement)
    'da_spill2': SP('state',1),  #Amount spilled out (displacement)
    'size_srcmouth': SP('state',1),  #Size of mouth of the source container
    }
  domain.Models={
    'Fmvtopour2': [  #Move to pouring point
      ['p_pour_trg'],
      ['lp_pour'],None],
    # 'Fmvtopour2': [  #Move to pouring point
    #   ['p_pour_trg','size_srcmouth','shake_axis2'],
    #   ['da_pour','da_spill2'],None],
    'Fflowc_tip10': [  #Flow control with tipping.
      ['lp_pour','size_srcmouth'],
      ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    'Fflowc_shakeA10': [  #Flow control with shake_A.
      ['lp_pour','size_srcmouth','shake_axis2'],
      ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    # 'Fflowc_tip10': [  #Flow control with tipping.
    #   ['lp_pour','size_srcmouth',
    #     "da_trg","a_src"],
    #   ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    # 'Fflowc_shakeA10': [  #Flow control with shake_A.
    #   ['lp_pour','size_srcmouth','shake_axis2',
    #     "da_trg","a_src"],
    #   ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    #  'Rdamount': [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
    #              TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2)]
      "Rdamount" : [['da_pour'],[REWARD_KEY],RwdModel()]
    }
  

  ###############################################################################

  mm= TModelManager(domain.SpaceDefs, domain.Models)
  mm.Load(LoadYAML(model_path+'model_mngr.yaml'), model_path)
  mm.Init()
  model = mm.Models[target_model][2]

  preds = defaultdict(list)
  for in_x in inputs:
    # pred = model.Predict(x=in_x, x_var=[0.0, 0.0], with_var=True)
    lppourx_var = mm.Models["Fmvtopour2"][2].Predict(x=[in_x[0]+0.6,in_x[2]+0.202], with_var=True).Var[0,0]
    pred = model.Predict(x=in_x, x_var=[lppourx_var, 0.0, 0.0, 0.0], with_var=True)
    # pred = model.Predict(x=in_x, x_var=[lppourx_var, 0.0, 0.0, 0.0, 0.0, 0.0], with_var=True)
    y = pred.Y[output_var_idx]
    y_var = pred.Var[output_var_idx,output_var_idx]
    preds["mean"].append(y)
    preds["sdv"].append(np.sqrt(y_var))
    preds["da_pour"].append(pred.Y[0])
    preds["da_pour_sdv"].append(np.sqrt(pred.Var[0,0]))

    # Rdamount = mm.Models["Rdamount"][2]
    Rdamount = TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2)
    # Rdamount.Load(data={"options": {"tune_h": True, "maxd1": 1e10, "maxd2": 1e10}})
    r = Rdamount.Predict(x=[pred.Y[0], 0.3, pred.Y[1], 0], x_var=[pred.Var[0,0], 0.0, pred.Var[1,1], 0.0], with_var=True)
    # r = Rdamount.Predict(x=[pred.Y[0]], x_var=[pred.Var[0,0]], with_var=True)
    preds["r"].append(r.Y.item())
    preds["r_sdv"].append(np.sqrt(r.Var.item()))

  for key in ["mean", "sdv", "da_pour", "da_pour_sdv", "r", "r_sdv"]:
    preds[key] = np.array(preds[key]).reshape(len(y_values), len(x_values))
  # print(preds)

  # pred = preds["mean"]
  # pred = preds["sdv"]
  pred = preds["r"]

  if False:
    fig = plt.figure(figsize=(20,4))
    plt.plot(np.linspace(0.065,0.075,len(pred[0])), pred[0].flatten())
    plt.subplots_adjust(left=0.05, right=0.95, top=0.8)
    plt.show()

  if False:
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=pred, x=x_values, y=y_values,
                              # colorscale='Oranges',
                              colorscale=[
                                [0.3, "rgb(1, 1, 1)"],
                                # [0, "rgb(1, 0, 0)"],
                                [0, "rgb(0, 0, 0)"],
                                [0.6, "rgb(1, 0, 0)"],
                                # [0.6, "rgb(1, 1, 1)"],
                              ],
                              zmin=0, zmax=0.6, zauto=False
                            ))
    fig.update_layout(height=800, width=800, title_text=fig_title+"<br>"+"min = "+str(round(pred.min(),3))+", max = "+str(round(pred.max(),3))+"<br><sub>"+subtitle+"<sub>", xaxis={"title": fig_xlabel}, yaxis={"title": fig_ylabel})
    fig.show()

    # fig = go.Figure()
    # fig.add_trace(go.Heatmap(z=preds["sdv"], x=x_values, y=y_values, colorscale='Oranges', zmin=0, zmax=0.2, zauto=False))
    # fig.update_layout(height=800, width=800, title_text=fig_title+"<br><sub>"+subtitle+"<sub>", xaxis={"title": fig_xlabel}, yaxis={"title": fig_ylabel})
    # fig.show()

  if True:
    tree_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"\
                +"bottomup/learn7/std_pour/ketchup/divided_in_six/plan/taylor_reward/best_est_trees"+"/"
    ests = defaultdict(list)
    for i in range(600,799+1):
      with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
        tree = joblib.load(f)
        node = None
        for key in tree.Tree.keys():
          if (target_model=="Fflowc_tip10" and key.A=="n4tir") or (target_model=="Fflowc_shakeA10" and key.A=="n4sar"):
            node = key
            # print(node)
            break
          # if (key.A=="n4tir") or (key.A=="n4sar"):
          #   print(key.A, tree.Tree[key].XS)
        # print(i)
        # print(tree.Tree[node].XS)
        smsz = tree.Tree[node].XS["size_srcmouth"].X.item()
        lp_pour_x = tree.Tree[node].XS["lp_pour"].X[0].item()
        r = tree.Tree[node].XS[".r"].X.item()
        ests["smsz"].append(smsz)
        ests["lp_pour_x"].append(lp_pour_x)
        ests["r"].append(r)
    # print(ests)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ests["lp_pour_x"], y= ests["smsz"],  
      # marker_color=data[name]["color"], 
      # name=name,
      mode='markers',
      # c=colors[i]
    ))
    fig.add_trace(go.Heatmap(z=-np.log10(-pred), x=x_values, y=y_values,
                              # colorscale='Oranges',
                              colorscale=[
                                # [0.3, "rgb(1, 1, 1)"],
                                # [0, "rgb(0, 0, 0)"],
                                # [0.6, "rgb(1, 0, 0)"],
                                [0, "rgb(255, 0, 0)"],
                                [0.33, "rgb(255, 125, 125)"],
                                [0.66, "rgb(255, 200, 125)"],
                                [1, "rgb(255, 255, 255)"],
                              ],
                              # zmin=0, zmax=0.6, zauto=False
                              # zmin=-0.5, zmax=0, zauto=False
                              colorbar=dict(
                                title="E[return]",
                                titleside="top",
                                tickmode="array",
                                tickvals=[2,1,0, -1],
                                ticktext=["-0.01", "-0.1", "-1", "-10"],
                                ticks="outside" 
                              )
                            ))
    fig.update_layout(height=800, width=800, title_text=fig_title+"<br>"+"min = "+str(round(pred.min(),3))+", max = "+str(round(pred.max(),3))+"<br><sub>"+subtitle+"<sub>", xaxis={"title": fig_xlabel}, yaxis={"title": fig_ylabel})
    fig.show()


  if False:
    name_list = []
    preds = []
    data = defaultdict(lambda: defaultdict(list))
    for i, (X, Y) in enumerate(zip(model.DataX, model.DataY)):
      # pred = model.Predict(x=X, x_var=0.0, with_var=True, with_grad=True)
      da_pour = Y[0]
      da_spill2 = Y[1]
      smsz = X[3]
      x, y = X[0], X[3]
      # print(X[2])

      if True:
            
        if not (
          -0.1<=x<=0 
          and round(X[2],3)==0.325
          and round(smsz,2)==0.07
          # and 0.30<X[2]<0.32
          # and 0.32<X[2]<0.33
          # and i>len(model.DataX)-400
        ):
          continue
        print(X[0])

        if da_pour < 0.3:
          if 0.2 <= da_pour:
            color = "rgba(255,0,255,1)"
            name = "0.2 <= da_pour < 0.3"
          elif 0.1 <= da_pour:
            color = "rgba(0,0,255,1)"
            name = "0.1 <= da_pour < 0.2"
          else:
            color = "rgba(0,0,0,1)"
            name = "0 <= da_pour < 0.1"
        else:
          if da_pour < 0.4:
            color = "rgba(255,127.5,63.75,1)"
            name = "0.3<= da_opur < 0.4"
          else:
            color = "rgba(255,0,0,1)"
            name = "0.4 <= da_pour <= 0.55"

        data[name]["x"].append(x)
        data[name]["y"].append(y)
        data[name]["color"] = color
        name_list.append(name)

    fig = go.Figure()
    for name in ["0.4 <= da_pour <= 0.55", "0.3<= da_opur < 0.4", "0.2 <= da_pour < 0.3", "0.1 <= da_pour < 0.2", "0 <= da_pour < 0.1"]:
      fig.add_trace(go.Scatter(x=data[name]["x"], y=data[name]["y"],  marker_color=data[name]["color"], name=name,
      mode='markers',
      # c=colors[i]
      ))
    fig.add_trace(go.Heatmap(z=pred, x=x_values, y=y_values,
                              # colorscale='Oranges',
                              # colorscale=[
                              #   [0.3, "rgb(1, 1, 1)"],
                              #   # [0, "rgb(1, 0, 0)"],
                              #   [0, "rgb(0, 0, 0)"],
                              #   [0.6, "rgb(1, 0, 0)"],
                              #   # [0.6, "rgb(1, 1, 1)"],
                              # ],
                              zmin=-2, zmax=0, zauto=False,
                            ))
    fig.update_layout(height=800, width=800, title_text=fig_title+"<br>"+"min = "+str(round(pred.min(),3))+", max = "+str(round(pred.max(),3))+"<br><sub>"+subtitle+"<sub>", xaxis={"title": fig_xlabel}, yaxis={"title": fig_ylabel},
                        legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=1.15)
                      )
    fig.show()

  if False:
    ests = defaultdict(list)
    for i in range(len(mm.Models["Fmvtopour2"][2].DataX)):
      with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
        tree = joblib.load(f)
        node = None
        for key in tree.Tree.keys():
          if (target_model=="Fflowc_tip10" and key.A=="n4tir") or (target_model=="Fflowc_shakeA10" and key.A=="n4sar"):
            node = key
            # print(node)
            break
        # print(i)
        # print(tree.Tree[node].XS)
        smsz = tree.Tree[node].XS["size_srcmouth"].X.item()
        lp_pour_x = tree.Tree[node].XS["lp_pour"].X[0].item()
        r = tree.Tree[node].XS[".r"].X.item()
        ests["smsz"].append(smsz)
        ests["lp_pour_x"].append(lp_pour_x)
        ests["r"].append(r)

    name_list = []
    preds = []
    data = defaultdict(lambda: defaultdict(list))
    for i in range(len(mm.Models["Fmvtopour2"][2].DataX)):

      if True:
            
        if not (
          i>=len(mm.Models["Fmvtopour2"][2].DataX)-10
          # and 0.30<X[2]<0.32
          # and 0.32<X[2]<0.33
          # and i>len(model.DataX)-400
        ):
          continue

        name = "optimized point"
        x = ests["lp_pour_x"][i]
        y = ests["smsz"][i]
        color = "black"
        data[name]["x"].append(x)
        data[name]["y"].append(y)
        data[name]["color"] = color

    fig = go.Figure()
    for name in ["optimized point"]:
      fig.add_trace(go.Scatter(x=data[name]["x"], y=data[name]["y"],  marker_color=data[name]["color"], name=name,
      mode='markers',
      # c=colors[i]
      ))
    fig.add_trace(go.Heatmap(z=pred, x=x_values, y=y_values,
                              # colorscale='Oranges',
                              colorscale=[
                                [0.3, "rgb(1, 1, 1)"],
                                # [0, "rgb(1, 0, 0)"],
                                [0, "rgb(0, 0, 0)"],
                                [0.6, "rgb(1, 0, 0)"],
                                # [0.6, "rgb(1, 1, 1)"],
                              ],
                              zmin=0.0, zmax=0.6, zauto=False,
                            ))
    fig.update_layout(height=800, width=800, title_text=fig_title+"<br>"+"min = "+str(round(pred.min(),3))+", max = "+str(round(pred.max(),3))+"<br><sub>"+subtitle+"<sub>", xaxis={"title": fig_xlabel}, yaxis={"title": fig_ylabel},
                        legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=1.15)
                      )
    fig.show()

  if False:
    ests = defaultdict(list)
    for i in range(len(mm.Models["Fmvtopour2"][2].DataX)):
      with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
        tree = joblib.load(f)
        node = None
        for key in tree.Tree.keys():
          if (target_model=="Fflowc_tip10" and key.A=="n4tir") or (target_model=="Fflowc_shakeA10" and key.A=="n4sar"):
            node = key
            # print(node)
            break
        # print(i)
        # print(tree.Tree[node].XS)
        smsz = tree.Tree[node].XS["size_srcmouth"].X.item()
        lp_pour_x = tree.Tree[node].XS["lp_pour"].X[0].item()
        da_pour = tree.Tree[node].XS["da_pour"]
        r = tree.Tree[node].XS[".r"]
        ests["smsz"].append(smsz)
        ests["lp_pour_x"].append(lp_pour_x)
        ests["da_pour"].append(da_pour.X.item())
        ests["da_pour_sdv"].append(np.sqrt(da_pour.Cov.item()))
        ests["r"].append(r.X.item())
        ests["r_sdv"].append(np.sqrt(r.Cov.item()))

    x_list = []
    y_list = []
    y_list2 = []
    y_list3 = []
    data = defaultdict(lambda: defaultdict(list))
    for i in range(len(mm.Models["Fmvtopour2"][2].DataX)):
          
      if True:
            
        if not (
          i>=len(mm.Models["Fmvtopour2"][2].DataX)-10
          # and 0.30<X[2]<0.32
          # and 0.32<X[2]<0.33
          # and i>len(model.DataX)-400
        ):
          continue

        x_list.append(ests["lp_pour_x"][i])
        y_list.append(ests["r"][i])
        y_list2.append(ests["r_sdv"][i])
        y_list3.append(ests["da_pour"][i])

    fig = plt.figure(figsize=(5.5,7))
    ax1 = fig.add_subplot(3,1,1)
    ax1.set_ylabel("E[r] +/- Std[r]")
    ax1.errorbar(x_values, preds["r"].reshape(-1,), yerr=preds["r_sdv"].reshape(-1,), capsize=2, ecolor='pink')
    # ax1.scatter(x_list, y_list, color="orange")
    ax1.set_ylim(-2,0.1)

    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
    ax2.plot(x_values, preds["r_sdv"].reshape(-1,))
    # ax2.scatter(x_list, y_list2, color="orange")
    ax2.set_ylabel("Std[r]")
    # ax2.set_yscale("log")
    ax2.set_ylim(0,0.6)

    ax3 = fig.add_subplot(3,1,3, sharex=ax1)
    ax3.set_ylabel("E[da_pour] +/- Std[da_pour]")
    ax3.errorbar(x_values, preds["da_pour"].reshape(-1,), yerr=preds["da_pour_sdv"].reshape(-1,), capsize=2, ecolor='pink', zorder=-1)
    # ax3.scatter(x_list, y_list3, color="orange", zorder=0)
    ax3.set_xlabel("lp_pour_x")

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.subplots_adjust(top=0.95, right=0.95)
    plt.show()