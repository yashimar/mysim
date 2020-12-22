from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import pandas as pd

def Help():
  pass

def Run(ct, *args):
  
  #set model_path, domain, target_model, inputs, output_var_idx

  #set model_path
  root_path = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'
  model_path = root_path + args[0] + "/models/"

  #set target_model
  target_model = "Fmvtopour2"

  #set inputs
  # x_values = np.linspace(0.38, 0.45, 20)    #p_pour_trg_x
  # y_values = np.linspace(0.18, 0.28, 20)[::-1]     #p_pour_trg_z
  x_values = np.linspace(0.2, 1.2, 80)    #p_pour_trg_x
  y_values = np.linspace(0.1, 0.3, 80)[::-1]     #p_pour_trg_z
  pc_rcv_x = 0.8
  inputs = []
  for y in y_values:
    for x in x_values:
      inputs.append([
        x,
        y,
        pc_rcv_x-0.15,
        -0.2,
        0.2025,
        pc_rcv_x+0.15,
        -0.2,
        0.2025,
        pc_rcv_x+0.15,
        0.2,
        0.2025,
        pc_rcv_x-0.15,
        0.2,
        0.2025,
      ])
  inputs = np.array(inputs)
  
  #set output_var_idx
  output_var_idx = 1  #da_spill2

  #set fig params
  fig_title = "leran2 nobounce's da_spill2 heatmap"
  subtitle = "smsz=" + str(0.065) + ", shake_axis2=(" + str(0.08) + "," + str(0.0) + ")"
  fig_xlabel = "p_pour_trg_x"
  fig_ylabel = "p_pour_trg_z"
  
  #set domain
  domain= TGraphDynDomain()
  SP= TCompSpaceDef
  domain.SpaceDefs={
    'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position (x,z)
    'shake_axis2': SP('action',2,min=[0.01,-0.5*math.pi],max=[0.1,0.5*math.pi]),  #Pouring skill parameter for 'shake_A'
    'ps_rcv': SP('state',12),  #4 edge point positions (x,y,z)*4 of receiver
    'lp_pour': SP('state',3),  #Pouring axis position (x,y,z) in receiver frame
    'da_pour': SP('state',1),  #Amount poured in receiver (displacement)
    'da_spill2': SP('state',1),  #Amount spilled out (displacement)
    'size_srcmouth': SP('state',1),  #Size of mouth of the source container
    }
  domain.Models={
    'Fmvtopour2': [  #Move to pouring point
      ['p_pour_trg'],
      ['lp_pour'],None],
    'Fflowc_shakeA10': [  #Flow control with shake_A.
      ['lp_pour','size_srcmouth','shake_axis2'],
      ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    }
  

  ###############################################################################

  mm= TModelManager(domain.SpaceDefs, domain.Models)
  mm.Load(LoadYAML(model_path+'model_mngr.yaml'), model_path)
  mm.Init()
  model = mm.Models[target_model][2]
  

  preds = defaultdict(list)
  for in_x in inputs:
    pred = model.Predict(in_x, with_var=True)
    preds["mean"].append(pred.Y[output_var_idx])
    preds["sdv"].append(pred.Var[output_var_idx,output_var_idx])
  preds["mean"] = np.array(preds["mean"]).reshape(len(y_values), len(x_values))
  preds["sdv"] = np.sqrt(np.array(preds["sdv"]).reshape(len(y_values), len(x_values)))
  # print(preds)

  fig = go.Figure()
  fig.add_trace(go.Heatmap(z=preds["mean"], x=x_values, y=y_values, colorscale='Oranges', zmin=-0.05, zmax=0.2, zauto=False))
  fig.update_layout(height=800, width=800, title_text=fig_title+"<br><sub>"+subtitle+"<sub>", xaxis={"title": fig_xlabel}, yaxis={"title": fig_ylabel})
  fig.show()

  # fig = go.Figure()
  # fig.add_trace(go.Heatmap(z=preds["sdv"], x=x_values, y=y_values, colorscale='Oranges', zmin=0, zmax=0.2, zauto=False))
  # fig.update_layout(height=800, width=800, title_text=fig_title+"<br><sub>"+subtitle+"<sub>", xaxis={"title": fig_xlabel}, yaxis={"title": fig_ylabel})
  # fig.show()