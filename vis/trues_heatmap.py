import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict
import yaml
from core_tool import *

def Help():
  pass

def Run(ct, *args):
  name_log = args[0]
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  sl_path = root_path + name_log + "/sequence_list.yaml"
  
  with open(sl_path, "r") as yml:
    sl = yaml.load(yml)

  # trues = [sl[ep]["reward"][1][2][0]["da_spill2"] for ep in range(len(sl))]
  trues = [sl[ep]["reward"][1][3][0]["da_pour"] for ep in range(len(sl))]
  
  x_values = np.linspace(0.3, 0.45, 10)
  y_values = np.linspace(0.35, 0.45, 10)
  trues = np.array(trues).reshape((len(x_values), len(y_values))).T

  fig_title = "true bounce's da_spill2 heatmap" + "<br>" \
              + "min = " + str(round(trues.min(),2)) + ", max = " + str(round(trues.max(),2))
  # subtitle = "smsz = 0.03, shake_axis2 = (0.01, 0.0), " \
  #            + "referenced p_pour_trg = (0.43, 0.15)"
  subtitle = "lp_pour_z = 0.31"
  fig_xlabel = "shake_axis2_dim1"
  fig_ylabel = "p_pour_trg_dim1"

  fig = go.Figure()
  fig.add_trace(go.Heatmap(z=trues, x=x_values, y=y_values, colorscale='Oranges',
                            # zmin=0.0, zmax=0.3, zauto=False
                          ))
  fig.update_layout(height=800, width=800, title_text=fig_title+"<br><sub>"+subtitle+"<sub>", xaxis={"title": fig_xlabel}, yaxis={"title": fig_ylabel})
  fig.show()