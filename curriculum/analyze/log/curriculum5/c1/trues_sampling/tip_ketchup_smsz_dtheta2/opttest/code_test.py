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
from .learn2 import *
import time
from glob import glob


RFUNC = "r_func"


def Help():
    pass


def Run(ct, *args):
    path = lambda name: "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/{}/pred_true_log.yaml".format(name)
    Y = []
    for name in ["g0", "spd06g0", "spd04g0", "spd02g0"]:
        y = []
        with open(path(name), "r") as yml:
            log = yaml.load(yml)
        for i in range(100):
            y.append(log[i]["Fshake_amount"]["true_output"][0])
        Y.append(y)
    Y = np.array(Y)
    np.save("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal3.npy", Y)
    