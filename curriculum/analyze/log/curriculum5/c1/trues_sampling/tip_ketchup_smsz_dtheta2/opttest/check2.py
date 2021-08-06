#coding: UTF-8
from core_tool import *
from ay_py.core import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from scipy.stats import multivariate_normal
from .......util import *
from ........tasks_domain.util import Rmodel
from ..greedyopt import *
from copy import deepcopy
import dill
from .setup import *
from .learn2 import *
import time
from glob import glob


RFUNC = "r_func"


def Help():
    pass


def Run(ct, *args):
    name = "GMM6Sig003_LCB1/t1"
    if len(args) == 1: name = args[0]
    
    save_img_dir = PICTURE_DIR + "opttest/onpolicy/{}/".format(name) 
    logdir = BASE_DIR + "opttest/logs/onpolicy/{}/".format(name)
    if "Er" in name:
        dm = Domain.load(logdir+"dm.pickle")
        datotal = setup_datotal(dm, logdir)
        reward = setup_reward(dm, logdir)
        gmm = GMM6(dm.nnmodel, diag_sigma=[(1.0-0.1)/33.3, (0.8-0.3)/33.3], Gerr = 1.0)
        gmm.train(dm.log["true_r_at_est_opt_dthtea2"])
    else:
        dm = Domain2.load(logdir+"dm.pickle")
        datotal = setup_datotal(dm, logdir)
        reward = setup_reward2(dm, logdir)
        gmm = dm.gmm
        gmm.train(dm.log["true_r_at_est_opt_dthtea2"])

    X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ])
    gr = gmm.predict(X).reshape(100,100)
    er = reward[Er]
    sr = reward[Sr]
    ev = er - 1*(sr + gr)
    

    trace = defaultdict(list)
    for smsz_idx, smsz in enumerate(dm.smsz):
        trace[0].append(go.Scatter(
            x=dm.dtheta2, y=er[:,smsz_idx],
            mode='lines', 
            name="E[r] - SD[r] - 報酬飛び値予測",
            line=dict(color="red", dash="dash"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[0]*len(dm.dtheta2),
                arrayminus=(sr+gr)[:,smsz_idx],
                color="red",
                thickness=1.5,
                width=3,
            )
        ))
        trace[1].append(go.Scatter(
            x=dm.dtheta2, y=er[:,smsz_idx],
            mode='lines', 
            name="E[r] - SD[r]",
            line=dict(color="orange", dash="dash"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[0]*len(dm.dtheta2),
                arrayminus=sr[:,smsz_idx],
                color="orange",
                thickness=1.5,
                width=3,
            )
        ))
        for i,addv in enumerate(range(0,1)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                trace[2+i].append(go.Scatter(
                    x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TRUE][:,smsz_idx+addv]],
                    mode='markers', 
                    name="Unobs {:.3f}".format(tmp_smsz),
                    marker=dict(
                                color= "blue" if addv == 0 else "grey", 
                                size=8,
                                symbol="x",
                            ),
                    visible=False,
                ))
            else:
                trace[2+i].append(go.Scatter(x=[], y=[]))
        for i,addv in enumerate(range(-4,5)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == tmp_smsz]
                    trace[3+i].append(go.Scatter(
                            x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["true_datotal"])[log_smsz_idx_list]],
                            mode='markers', 
                            name="Obs {:.3f}".format(tmp_smsz),
                            marker=dict(
                                color= "purple" if addv == 0 else "pink", 
                                size=8,
                            ),
                            visible=False,
                    ))
                else:
                    trace[3+i].append(go.Scatter(x=[], y=[]))
            else:
                trace[3+i].append(go.Scatter(x=[], y=[]))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for smsz_idx, smsz in enumerate(dm.smsz):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(dm.smsz)
            trace["vis{}".format(j)][smsz_idx] = True
        step = dict(
            method="update",
            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                {"title": "size_srcmouth: {:.4f}".format(smsz)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "size_srcmouth: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta2"
    fig['layout']['yaxis']['title'] = "return"
    fig['layout']['yaxis']['range'] = (-8,0.5)
    for smsz_idx, smsz in enumerate(dm.smsz):
        fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "return2.html", auto_open=False)
            