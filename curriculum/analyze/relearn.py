# from core_tool import *
from util import *
from ..tasks_domain.detach_datotal.pouring import task_domain as td


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum3/pouring/full_scratch/curriculum_test/t1/first300"
    save_path = "curriculum3/pouring/relearn/addone"
    model_name = "Ftip_amount"
    pref = ""
    
    model, DataX, DataY = remake_model(td, model_name, model_path, save_path, pref)
    print(model.Options)
    for x,y in zip(DataX, DataY):
        x[2] += 1.
        x[8] += 1.
        model.Update(x.tolist(),y.tolist(),not_learn=False)
    
    check_or_create_dir(ROOT_PATH+save_path)
    with open(ROOT_PATH+save_path+"/{}_{}.pkl".format(model_name, pref), "wb") as f:
        pickle.dump(model, f)