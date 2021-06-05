from core_tool import *
from util import *
from ..tasks_domain.pouring import task_domain as td

def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum2/pouring/full_scratch/curriculum_test/t1/first300"
    model_name = "Ftip"
    
    model = ModelManager(td.Domain(), ROOT_PATH+model_path).Models[model_name][2]
    mmsX_transition = model.DebugTrainingData()
    lppourx = mmsX_transition[1]
    lppoury = mmsX_transition[2]
    lppourz = mmsX_transition[3]
    m = mmsX_transition[8]
    
    plt.close()
    for i in [50,100,150,200]:
        fig = plt.figure()
        plt.hist(lppoury[i])
        plt.show()