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

    p = 3
    lam = 1e-5
    Var = np.diag([(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)])**2
    gmm = GMM5(dm.nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], lam = lam)
    # gmm = GMM4(dm.nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)])
    gmm.extract_jps()
    gmm.jumppoints.update({"X": [[0., x[1]] for x in gmm.jumppoints["X"]]})
    tx = np.array([x[1] for x in gmm.jumppoints["X"]])
    uq = np.unique(tx, return_index=True)[1][[0, 3, 7, 10, 15, 21, 22, 23]]
    tx = np.array([x[1] for x in gmm.jumppoints["X"]])
    gmm.jumppoints.update({"X": [x for i, x in enumerate(gmm.jumppoints["X"]) if i in uq]})
    gmm.jumppoints.update({"Y": [y for i, y in enumerate(gmm.jumppoints["Y"]) if i in uq]})
    gmm.train(recreate_jp = False)
    # for x, y, gc, w in zip(gmm.jumppoints["X"], gmm.jumppoints["Y"], gmm.gc_concat, gmm.w_concat):
    #     print(y, gmm.predict(x), w)
    
    x_list = np.linspace(0.3, 0.8, 1000)
    X = np.array([[0., x] for x in x_list])
    P = gmm.predict(X)
    
    ttx = np.array(gmm.jumppoints["X"])
    tty = np.array(gmm.jumppoints["Y"])
    
    jpmeta = [[(multivariate_normal.pdf([0.,x],jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy).item() for x in x_list] for jpx, jpy in zip(ttx, tty)]
    # jpx = ttx[0]
    # jpy = tty[0]
    # for x in x_list:
    #     tmp = multivariate_normal.pdf([0., x],jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy
    #     tmp = tmp.item()
    #     print(x, tmp)
    # # print(jpmeta[0])
    # print(jpx)
    # jpge
    
    tttx = [x[1] for x in ttx]
    fig = plt.figure()
    for jp in jpmeta:
        plt.scatter(x_list, jp, c="skyblue")
    plt.scatter(x_list, P, c="purple")
    plt.scatter(tttx, tty, c="orange")
    # plt.scatter(tttx, gmm.predict(ttx))
    plt.show()