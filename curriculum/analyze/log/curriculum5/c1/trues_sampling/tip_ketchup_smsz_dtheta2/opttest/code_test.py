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


RFUNC = "r_func"


def Help():
    pass


def Run(ct, *args):
    name = "t0.1/t1"
    
    logdir = BASE_DIR + "opttest/logs/{}/".format(name)
    dm = Domain.load(logdir+"dm.pickle")
    observations = np.array([
        dm.log["est_opt_dtheta2"],
        dm.log["smsz"]
    ]).T
    
    obsr = ObservationReward(observations)
    obsr.setup(diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50])
    x = [0.58, 0.45]
    print(obsr.calc_reward(x))