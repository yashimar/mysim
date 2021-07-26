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
    pref = "Er"
    name = "onpolicy/Er/t25"
    # name = "onpolicy/GMM4Sig003_gnnsd1_ggmm2_LCB2/t12"
    
    logdir = BASE_DIR + "opttest/logs/{}/".format(name)
    dm = Domain.load(logdir+"dm.pickle")

    # gmm = GMM4(dm.nnmodel, diag_sigma=[(1.0-0.1)/33.3, (0.8-0.3)/33.3], Gerr = 1.0)
    gmm_name_list = []
    unobs_name_list = []
    
    datotal = setup_datotal(dm, logdir)
    gmmpred = setup_gmmpred(dm, gmm_name_list, logdir)
    unobspred = setup_unobssd(dm, unobs_name_list, logdir)
    reward = setup_reward(dm, logdir)
    
    opt_dtheta2_list = np.argmax(reward[pref], axis = 0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dm.smsz, y=[smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[RFUNC].T, opt_dtheta2_list))],
            mode='markers', 
            name="{}".format(pref),
        )
    )
    fig.show()