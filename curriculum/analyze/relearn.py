from core_tool import *
from util import *
from ..tasks_domain.flow_ctrl import task_domain as td


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5_5_5"
    save_path = "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/relearn"
    model_name = "Ftip"
    suff = "amp_smsz_dtheta2"
    
    model, DataX, DataY = remake_model(td, model_name, model_path, save_path, suff)
    print(model.Options)
    for x,y in zip(DataX, DataY):
        x[5] *= 10
        x[11] *= 100
        model.Update(x.tolist(),y.tolist(),not_learn=False)
    
    check_or_create_dir(ROOT_PATH+save_path)
    with open(ROOT_PATH+save_path+"/{}_{}.pkl".format(model_name, suff), "wb") as f:
        pickle.dump(model, f)