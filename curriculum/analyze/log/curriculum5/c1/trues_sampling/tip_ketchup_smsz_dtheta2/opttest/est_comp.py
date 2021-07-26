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
    pref_rname_list = [
        ("Er", "Er"),
        ("Er_LCB2", "Er_LCB2"),
        ("GMM4Sig003_gnnsd1_ggmm1", "gmm_gnnsd1.0_ggmm1.0~Er"),
        ("GMM4Sig003_gnnsd1_ggmm1_LCB2", "gmm_gnnsd1.0_ggmm1.0~Er_LCB2"),
        ("GMM4Sig003_gnnsd1_ggmm2", "gmm_gnnsd1.0_ggmm2.0~Er"),
        ("GMM4Sig003_gnnsd1_ggmm2_LCB2", "gmm_gnnsd1.0_ggmm2.0~Er_LCB2"),
        ("GMM4Sig005_gnnsd1_ggmm1", "gmm_gnnsd1.0_ggmm1.0~Er"),
        ("GMM4Sig005_gnnsd1_ggmm1_LCB2", "gmm_gnnsd1.0_ggmm1.0~Er_LCB2"),
        ("GMM4Sig003_gnnsd1_ggmm3", "gmm_gnnsd1.0_ggmm3.0~Er"),
        ("GMM4Sig005_gnnsd1_ggmm2", "gmm_gnnsd1.0_ggmm2.0~Er"),
    ]
    trial_list = ["t{}".format(i) for i in range(2,18)]
    
    opter_list_meta = []
    opttrue_list_meta = []
    for pref, rname in pref_rname_list:
        save_img_dir = PICTURE_DIR + "opttest/onpolicy/{}/".format(pref)
        name_list = ["onpolicy/{}/{}".format(pref,trial) for trial in trial_list]
        opter_list = []
        opttrue_list = []
        for name in name_list:
            logdir = BASE_DIR + "opttest/logs/{}/".format(name)
            t = time.time()
            dm = Domain.load(logdir+"dm.pickle")

            setup_datotal(dm, logdir)
            gmm_name_list = [None if dm.gmm == None else (dm.gmm, "gmm")]
            setup_gmmpred(dm, gmm_name_list, logdir)
            setup_unobssd(dm, [], logdir)
            gmm_names = [gmm_name for _, gmm_name in gmm_name_list]
            try:
                gp = dm.gain_pairs
            except: 
                gp = (1.0 ,1.0)
            gain_pairs = [gp]
            reward = setup_reward(dm, logdir, gmm_names, gain_pairs)

            opt_dtheta2_list = np.argmax(reward[rname], axis = 0)
            opter_list.append([smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(reward[rname].T, opt_dtheta2_list))])
            opttrue_list.append([smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[RFUNC].T, opt_dtheta2_list))])
        opter_list_meta.append(opter_list)
        opttrue_list_meta.append(opttrue_list)
    
    
    fig = go.Figure()
    for opter_list, (pref, _) in zip(opter_list_meta, pref_rname_list):
        fig.add_trace(
            go.Scatter(
                x=dm.smsz, y=np.mean(opter_list, axis=0),
                mode='markers', 
                name="{}".format(pref),
                error_y=dict(
                        type="data",
                        symmetric=True,
                        array=np.std(opter_list, axis=0),
                        thickness=1.5,
                        width=3,
                    )
            )
        )
    fig['layout']['yaxis']['range'] = (-6,0.1)
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "est reward"
    plotly.offline.plot(fig, filename = PICTURE_DIR + "opttest/onpolicy/" + "estr_comp.html", auto_open=False)