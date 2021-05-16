from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from util import *
from ..tasks_domain.util import Rmodel
from ..tasks_domain.flow_ctrl import task_domain as td


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5_5_5"
    save_dir = PICTURE_DIR + model_path.replace("/","_") + "/"
    file_name_pref = ""
    
    model_name = "Ftip"
    xs_value = {
        "gh_abs": [0.25],
        "lp_pour_x": [0.],
        "lp_pour_y": [0.],
        "lp_pour_z": [0.],
        "da_trg": [0.3],
        "size_srcmouth": [0.055],
        "material2": [0.1, 0.2, 0.0, 0.1],
        "dtheta1": [1.4e-2],
        "dtheta2": [0.002],
    }
    input_features = ["gh_abs","lp_pour_x","lp_pour_y","lp_pour_z","da_trg","size_srcmouth","material2","dtheta1","dtheta2"]
    X = {"feature": "lp_pour_x", "values": np.linspace(-0.5,0.7,40)}
    Y = {"feature": "lp_pour_z", "values": np.linspace(-0.2,0.6,40)}
    z = {"feature": "da_total", "output_dim": 0, "range": {MEAN: [-0.05,0.8], SIGMA: [-0.05,0.35]}}
    reward_function = {
        "name": "Rdatotal",
        "model": Rmodel("Fdatotal_gentle"),
        "input_features": ["da_trg","da_total"],
        "format_mean": lambda pred: [pred.Y[0]],
        "format_var": lambda pred: [pred.Var[0,0]],
        "range": {MEAN: [-3.,0.], SIGMA: [-0.05,2.0]}
    }
    
    plot_dynamics_heatmap(td, model_path, save_dir, file_name_pref, model_name, xs_value, input_features, X, Y, z, reward_function)
