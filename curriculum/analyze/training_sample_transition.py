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
    lppourz = mmsX_transition[3]
    
    # print(lppourx[-1])
    # print(model.MmsX().T[1])
    print(all(np.array(lppourx[-1])-np.array(model.MmsX().T[1])<1e-4))
    # print(lppourx[-1][-10:], model.MmsX().T[1][-10:])
    # plt.close()
    # plt.hist(lppourx[50])
    # plt.show()