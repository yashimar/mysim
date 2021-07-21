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
    name = "onpolicy/Er/t2"
    # name = "t0.1/2000/t2v2"
    
    logdir = BASE_DIR + "opttest/logs/{}/".format(name)
    t = time.time()
    dm = Domain.load(logdir+"dm.pickle")
    print(dm.nnmodel.basedir)
    dm.nnmodel.modeldir = logdir + "{}/".format("models")
    dm.nnmodel.setup()
    dm.nnmodel.hogehoge = "hogehoge"
    
    print(dm.nnmodel.hogehoge)
    print(dm.nnmodel.basedir)
    print(len(dm.log["ep"]))
    print(len(dm.nnmodel.model.DataX))
    # traeod = dm.log["true_r_at_est_opt_dthtea2"]
    # smsz = dm.log["smsz"]
    # y = [t if s<=0.65 else None for t,s in zip(traeod,smsz)]
    
    # fig = plt.figure()
    # plt.scatter(dm.log["ep"], y)
    # plt.show()
    