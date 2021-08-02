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
    pref_list = [
        "Er",
        # "Er_LCB2",
        # "GMM4Sig003_gnnsd1_ggmm1",
        # "GMM4Sig003_gnnsd1_ggmm1_LCB2",
        # "GMM4Sig003_gnnsd1_ggmm2",
        # "GMM4Sig003_gnnsd1_ggmm2_LCB2",
        # "GMM4Sig003_gnnsd1_ggmm3",
        # "GMM4Sig005_gnnsd1_ggmm1",
        # "GMM4Sig005_gnnsd1_ggmm1_LCB2",
        # "GMM4Sig005_gnnsd1_ggmm2",
        # "GMM5Sig003_gnnsd1_ggmm1",
        "GMM6Sig003_LCB1",
        "GMM6Sig001_LCB1",
    ]
    n_ep = 500
    smsz_thr = 0.65
    num = 50
    trial_list = ["t{}".format(i) for i in range(1,31)]

    ymean_list_meta = []
    barset = []
    for pref in pref_list:
        save_img_dir = PICTURE_DIR + "opttest/onpolicy/{}/".format(pref)
        name_list = ["onpolicy/{}/{}".format(pref,trial) for trial in trial_list]
        y_list_meta = [[] for _ in range(n_ep)]
        yvis_list_meta = [[] for _ in range(n_ep)]
        smsz_list_meta = [[] for _ in range(n_ep)]
        for name in name_list:
            logdir = BASE_DIR + "opttest/logs/{}/".format(name)
            if not os.path.exists(logdir+"dm.pickle"):
                continue
            t = time.time()
            dm = Domain.load(logdir+"dm.pickle")

            traeod = dm.log["true_r_at_est_opt_dthtea2"]
            smsz = dm.log["smsz"]
            for i,(t,s) in enumerate(zip(traeod,smsz)):
                if s <= smsz_thr:
                    yvis_list_meta[i].append(t)    
                y_list_meta[i].append(t)
                smsz_list_meta[i].append(s)
  
        # yvis_list_meta = [np.where(np.array(yi_list)<-1, -1, yi_list) for yi_list in yvis_list_meta]
        ymean_list = []
        ysd_list = []
        for yi_list in yvis_list_meta:
            ymean_list.append(np.mean(yi_list))
            ysd_list.append(np.std(yi_list))
        ymean_list_meta.append(ymean_list)
        smsz = lambda ep,smsz_list_meta=smsz_list_meta: np.hstack([smsz_list for smsz_list in np.array(smsz_list_meta)[ep].T])
        # y_list_meta = [np.where(np.array(yi_list)<-1, -1, yi_list) for yi_list in y_list_meta]
        y = lambda ep,y_list_meta=y_list_meta: np.hstack([y_list for y_list in np.array(y_list_meta)[ep].T])
        barset.append((smsz,y))
        

        # ep = range(0,500)
        # plt.bar(left=smsz(ep), height=y(ep), width=0.03)
        # plt.show()
        # hoge

        
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

    
    ymean_mva_list = [pd.Series(ymean_list).rolling(num).mean() for ymean_list in ymean_list_meta]
    ymean_std_list = [pd.Series(ymean_list).rolling(num).std() for ymean_list in ymean_list_meta]

    fig = go.Figure()
    for ymean_mva, ymean_std, pref in zip(ymean_mva_list, ymean_std_list, pref_list):
        fig.add_trace(go.Scatter(
            x=np.linspace(0,n_ep-1,n_ep), y=ymean_mva,
            mode='lines', 
            name=pref,
            error_y=dict(
                    type="data",
                    symmetric=False,
                    array=np.zeros(len(ymean_std)),
                    arrayminus=ymean_std,
                    thickness=1.5,
                    width=3,
                )
        ))
    fig['layout']['yaxis']['range'] = (-3,0.1)
    fig['layout']['xaxis']['title'] = "episode"
    fig['layout']['yaxis']['title'] = "reward"
    fig['layout']['title'] = "window size: {}, smsz<{}".format(num, smsz_thr)
    plotly.offline.plot(fig, filename = PICTURE_DIR + "opttest/onpolicy/" + "reward_comp.html", auto_open=False)    
    
    
    for ep in [range(0,500), range(100), range(0,200), range(0,300), range(0,500), range(100,200), range(200,300), range(300,400), range(400,500)]:
        fig = go.Figure()
        for (smsz,y), pref in zip(barset, pref_list):
            idx = np.argsort(smsz(ep))
            xt = smsz(ep)[idx]
            yt = y(ep)[idx]
                    
            ms = []
            sds = []
            for s in dm.smsz:
                idx = [i for i, xtt in enumerate(xt) if xtt == s]
                ms.append(np.mean(yt[idx]))
                sds.append(np.std(yt[idx]))
                
            fig.add_trace(go.Scatter(x = dm.smsz, y = ms,
                name=pref,
                mode="markers",
                marker=dict(
                    size = 8,
                    line=dict(
                        width=1,
                        color="gray",
                    )
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=np.zeros(len(sds)),
                    arrayminus=sds,
                    thickness=3,
                    width=6,
                ),
            ))
            # fig.add_trace(go.Scatter(x = dm.smsz, y = ms,
            #     name=pref,
            #     mode="lines",
            #     line=dict(
            #         dash='dot',
            #         width=1,
            #     ),
            # )) 

        fig['layout']['yaxis']['range'] = (-8,0.1)
        fig['layout']['xaxis']['title'] = "size_srcmouth"
        fig['layout']['yaxis']['title'] = "reward"
        fig['layout']['title'] = "ep{} ~ {}".format(ep[0], ep[-1])
        plotly.offline.plot(fig, filename = PICTURE_DIR + "opttest/onpolicy/epreward/" + "ep{}ep{}.html".format(ep[0], ep[-1]), auto_open=False)
        
        
    
