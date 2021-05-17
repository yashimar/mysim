from core_tool import *
from util import *
from ..tasks_domain.flow_ctrl import task_domain as td


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5_5_5"
    model_name = "Ftip"
    
    mm = ModelManager(td.Domain(), ROOT_PATH+model_path)
    model = mm.Models[model_name][2]
    model.Params["nn_params"], model.Params["nn_params_err"] = None, None
    model.Init()
    
    pred_test(model)