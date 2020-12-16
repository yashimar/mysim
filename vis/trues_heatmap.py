import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
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

  trues = [sl[ep]["reward"][3][2][0]["da_spill2"] for ep in range(len(sl))]
  
  x_values = np.linspace(0.38, 0.45, 20)
  y_values = np.linspace(0.18, 0.28, 20)
  trues = np.array(trues).reshape((len(x_values), len(y_values))).T

  fig_title = "true nobounce's da_spill2 heatmap"
  subtitle = "smsz=" + str(0.055) + ", shake_axis2=(" + str(0.1) + "," + str(0.25) + "), " + "gh_ratio=" + str(0.5)
  fig_xlabel = "lp_pour_x"
  fig_ylabel = "lp_pour_z"

  fig = go.Figure()
  fig.add_trace(go.Heatmap(z=trues, x=x_values, y=y_values, colorscale='Oranges', zmin=0, zmax=0.2, zauto=False))
  fig.update_layout(height=800, width=800, title_text=fig_title+"<br><sub>"+subtitle+"<sub>", xaxis={"title": fig_xlabel}, yaxis={"title": fig_ylabel})
  fig.show()