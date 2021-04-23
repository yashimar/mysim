from matplotlib import pyplot as plt
from ..vis.util import FormatSequence
from ..tasks_domain.util import ModelManager
from ..tasks_domain import pouring as td
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')

ROOT_PATH = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'


def Help():
    pass


def Run(ct, *args):
    path = "curriculum/manual_skill_ordering2/ketchup_0055/first"
    envs, trues, ests = FormatSequence(path)
    domain = td.Domain()
    mm = ModelManager(domain, ROOT_PATH+path)
    p_pour_trg = np.array([[x[1], x[2]] for x in mm.Models["Fmvtopour2"][2].DataX]).reshape(-1, 2)
    dtheta2 = [x[-1] for x in mm.Models["Fflowc_tip10"][2].DataX]

    x = p_pour_trg[:,0]
    y = p_pour_trg[:,1]

    fig = plt.figure(figsize=(20, 6))
    ts = [0, 11, 19, 27, 35, 43, 51, 59, len(x)]
    for i in range(1, len(ts)):
        es = ts[i-1]
        ee = ts[i]
        for j in range(es, ee):
            c = "blue" if trues[".r"][j] > -1 else "red"
            fig.add_subplot(2, len(ts)/2, i).scatter(x=x[j], y=y[j], s=30, alpha=0.5, c=c)
        plt.xlim(0.1, 1.3)
        plt.ylim(0.05, 0.75)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.8)
    plt.show()
