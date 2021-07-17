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
    name = "onpolicy/Er/t4"
    name_list = [
        "onpolicy/Er/t1",
        "onpolicy/Er/t2",
        "onpolicy/Er/t3",
        "onpolicy/Er/t4",
        "onpolicy/Er/t5",
        "onpolicy/Er/t6",
        "onpolicy/Er/t7",
        "onpolicy/Er/t8",
        "onpolicy/Er/t9",
        "onpolicy/Er/t10",
    ]
    for name in name_list:
        logdir = BASE_DIR + "opttest/logs/{}/".format(name)
        t = time.time()
        dm = Domain.load(logdir+"dm.pickle")
    
    traeod = dm.log["true_r_at_est_opt_dthtea2"]
    smsz = dm.log["smsz"]
    y = [t if s<=0.65 else None for t,s in zip(traeod,smsz)]
    
    fig = plt.figure()
    plt.scatter(dm.log["ep"], y)
    plt.show()
    