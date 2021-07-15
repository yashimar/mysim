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
    name = "t0.1/2000/t2v2"
    
    logdir = BASE_DIR + "opttest/logs/{}/".format(name)
    dm = Domain.load(logdir+"dm.pickle")
    observations = np.array([
        dm.log["est_opt_dtheta2"],
        dm.log["smsz"]
    ]).T
    
    gmm = GMM2(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
    gmm.train()
    
    x = [
        [0.3101, 0.52727],
        [0.2101, 0.6127],
    ]
    print(gmm.predict(x))
    # fig = plt.figure()
    # y_list = []
    # for dtheta2 in dm.dtheta2:
    #     x = [dtheta2, 0.52727]
    #     y_list.append(us.calc_sd(x).item())
    # plt.plot(dm.dtheta2, y_list)
    # fig.show()
    