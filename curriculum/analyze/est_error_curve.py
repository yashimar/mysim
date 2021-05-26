from core_tool import *
from util import *
from ..tasks_domain.pouring import task_domain as td


def Help():
    pass


def Run(ct, *args):
    log_name_list = [
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c0_init_50",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c1_small_5",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c2_small_nobounce_tip_dtheta2_3",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c3_small_5",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c4_small_ketchup_5",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c5_middle_nobounce_5",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c6_middle_ketchup_5",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c7_large_nobounce_5",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5_5",
        # "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5_5_5",
        # "curriculum/pouring3/full_scratch/curriculum_test/t1/first150",
        "curriculum2/pouring/full_scratch/curriculum_test/t1/first300",
    ]
    save_dir = PICTURE_DIR + "curriculum2/pouring/full_scratch/curriculum_test/t1".replace("/","_") + "/"
    file_name_pref = ""
    dynamics_outdim_pair = {"Fmvtopour2": 3, "Ftip": 4, "Fshake": 4, "Famount": 2}
    vis_state_dynamics_outdim_lim_pair = [
        ("da_total_tip", "Ftip", 0, (-0.1,0.9)),
        ("lp_flow_x_tip", "Ftip", 1, (-0.3,1.5)),
        ("lp_flow_y_tip", "Ftip", 2, (-0.1,0.1)),
        ("flow_var_tip", "Ftip", 3, (0.1,1.0)),
        ("da_total_shake", "Fshake", 0, (-0.1,0.9)),
        ("lp_flow_x_shake", "Fshake", 1, (-0.3,1.5)),
        ("lp_flow_y_shake", "Fshake", 2, (-0.1,0.1)),
        ("flow_var_shake", "Fshake", 3, (0.1,1.0)),
        ("da_pour", "Famount", 0, (-0.1,0.9)),
        ("da_spill2", "Famount", 0, (-0.1,1)),
    ]
    
    go_layout = {
        'height': 800*2*len(vis_state_dynamics_outdim_lim_pair),
        'width': 1800,
        'margin': dict(t=30, b=0),
        'legend_tracegroupgap': 800-30,
        'hoverdistance': 5,
    }
    
    transition_plot(td, log_name_list, dynamics_outdim_pair, vis_state_dynamics_outdim_lim_pair, go_layout, save_dir, file_name_pref)