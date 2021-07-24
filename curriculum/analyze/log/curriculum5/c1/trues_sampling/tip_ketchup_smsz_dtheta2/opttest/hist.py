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
from .learn import *
import time
from glob import glob


RFUNC = "r_func"


def Help():
    pass


def Run(ct, *args):
    pref = "onpolicy/Er_LCB2"
    n_ep = 500
    smsz_thr = 0.63
    name_list = [
        # "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "t7",
        "t8",
        "t9",
        "t10",
        "t11",
        "t12",
        "t13",
        "t14",
        "t15",
        "t16",
        "t17",
        "t18",
        "t19",
        "t20",
        "t21",
        "t22",
        "t23",
        "t24",
    ]
    
    save_img_dir = PICTURE_DIR + "opttest/{}/".format(pref)
    name_list = ["{}/{}".format(pref,name) for name in name_list]
    y_list_meta = [[] for _ in range(n_ep)]
    yvis_list_meta = [[] for _ in range(n_ep)]
    smsz_list_meta = [[] for _ in range(n_ep)]
    for name in name_list:
        logdir = BASE_DIR + "opttest/logs/{}/".format(name)
        t = time.time()
        dm = Domain.load(logdir+"dm.pickle")

        traeod = dm.log["true_r_at_est_opt_dthtea2"]
        smsz = dm.log["smsz"]
        for i,(t,s) in enumerate(zip(traeod,smsz)):
            if s <= smsz_thr:
                yvis_list_meta[i].append(t)    
            y_list_meta[i].append(t)
            smsz_list_meta[i].append(s)
    
    ymean_list = []
    ysd_list = []
    for yi_list in yvis_list_meta:
        ymean_list.append(np.mean(yi_list))
        ysd_list.append(np.std(yi_list))
    
    text = ["<br />".join(["t{}: {} ({:.3f})".format(j+1,yij,sij) if yij >= -1 else "<b>t{}: {} ({:.3f})</b>".format(j+1,yij,sij) if sij <= smsz_thr else "t{}: {} ({:.3f}) ignore".format(j+1,yij,sij) for j,(yij,sij) in enumerate(zip(y_list_meta[i], smsz_list_meta[i]))]) for i in range(n_ep)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
                x=dm.log["ep"], y=ymean_list,
                mode='markers', 
                name="{}".format(name),
                text = text,
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=ysd_list,
                    thickness=1.5,
                    width=3,
                )
            )
        )
    fig['layout']['yaxis']['range'] = (-8,0.1)
    fig['layout']['xaxis']['title'] = "episode"
    fig['layout']['yaxis']['title'] = "reward"
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "reward.html", auto_open=False)


    # num = 100
    # ymean_mva = np.convolve(ymean_list, np.ones(num)/num, mode='same')
    # fig = plt.figure()
    # plt.plot(ymean_mva)
    # plt.show()