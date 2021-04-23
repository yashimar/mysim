from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from ..tasks_domain import pouring as td
from ..tasks_domain.util import ModelManager
from matplotlib import pyplot as plt
import plotly.graph_objects as go

ROOT_PATH = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum/manual_skill_ordering2/ketchup_0055/first"
    domain = td.Domain()
    mm = ModelManager(domain, ROOT_PATH+model_path)

    Fflowc_tip10 = mm.Models["Fflowc_tip10"][2]
    for x,t in zip(Fflowc_tip10.DataX, Fflowc_tip10.DataY):
        p = Fflowc_tip10.Predict(x, with_var=True)
        print("---")
        Print("input:", x)
        Print("dtheta2:", x[-1])
        Print("est:", p.Y[0].item())
        Print("sigma:", np.sqrt(p.Var[0,0]))
        Print("true:", t[0])
        Print("diff:", abs(p.Y[0].item()-t[0]))

    X = np.linspace(0.2-0.6,1.2-0.6,20)
    Y = np.linspace(0.002,0.02,20)
    inputs = []
    for y in Y:
        for x in X:
            inputs.append([0.25, x, 0., 0.27, 0.3, 5.5e-2, 1e-1, 1e-2, 2.5e-1, 2e-1, 1.4e-2, y])
    Z = []
    for input in inputs:
        p = Fflowc_tip10.Predict(input, with_var=True)
        Z.append(p.Y[0])
        # Z.append(np.sqrt(p.Var[0,0]))
        # print(input, p.Y[0])
    Z = np.array(Z).reshape(len(Y),len(X))

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Z, x=X, y=Y,
                              colorscale='Oranges',
                            #   zmin=0, zmax=0.55, zauto=False,
                              # zmin=-1, zmax=2.3, zauto=False,
                              # colorbar=dict(
                              #   title="E[return]",
                              #   titleside="top",
                              #   tickmode="array",
                              #   tickvals=[2, 1.3, 1, 0, -1],
                              #   ticktext=["-0.01", "-0.05", "-0.1", "-1", "-10"],
                              #   ticks="outside" 
                              # )
                            ))
    fig.update_layout(height=800, width=800, xaxis={"title": "lp_pour_x"}, yaxis={"title": "dtheta2"})
    fig.show()
