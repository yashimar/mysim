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
    name = "t0.1/2000/t2v2"
    
    logdir = BASE_DIR + "opttest/logs/{}/".format(name)
    dm = Domain.load(logdir+"dm.pickle")
    
    # reward = dict()
    # for fpath in glob(logdir+"/npydata/*"):
    #     name = (fpath.split("/")[-1]).split(".npy")[0]
    #     reward[name] = np.load(fpath)
    # print(reward["Er"].shape)

    with open(logdir+"datotal.pickle", mode="rb") as f:
        datotal = pickle.load(f)
    rmodel = Rmodel("Fdatotal_gentle")
    datotal_nnmean = datotal[NNMEAN]
    datotal_nnsd = datotal[NNSD]
    reward_normal = dict()
    if True:
        Print("reward_normal:", logdir)
        t = time.time()
        reaward_sm = np.array([[rmodel.Predict(x=[0.3, datotal_nnmean[idx_dtheta2, idx_smsz]], x_var=[0, datotal_nnsd[idx_dtheta2, idx_smsz]**2], with_var=True) for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        print(reaward_sm.shape)
        reward_normal[Er] = np.array([[reaward_sm[idx_dtheta2, idx_smsz].Y.item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        print(reward_normal[Er].shape)
        reward_normal[Sr] = np.sqrt([[reaward_sm[idx_dtheta2, idx_smsz].Var[0,0].item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        reward_normal[Er+"_"+LCB1] = reward_normal[Er] - 1*reward_normal[Sr]
        reward_normal[Er+"_"+LCB2] = reward_normal[Er] - 2*reward_normal[Sr]
        Print("Done:", time.time()-t)