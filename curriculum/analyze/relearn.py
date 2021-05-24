# from core_tool import *
from util import *
from ..tasks_domain.pouring import task_domain as td


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5_5_5"
    save_path = "test/mms4"
    model_name = "Ftip"
    suff = ""
    
    model, DataX, DataY = remake_model(td, model_name, model_path, save_path, suff)
    print(model.Options)
    for x,y in zip(DataX, DataY):
        model.Update(x.tolist(),y.tolist(),not_learn=False)
    
    check_or_create_dir(ROOT_PATH+save_path)
    with open(ROOT_PATH+save_path+"/{}_{}.pkl".format(model_name, suff), "wb") as f:
        pickle.dump(model, f)