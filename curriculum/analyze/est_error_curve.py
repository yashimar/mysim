# coding: UTF-8
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
    save_sh_dir = "curriculum2/pouring/full_scratch/curriculum_test/t1"
    file_name_pref = ""
    dynamics_iodim_pair = {"Fmvtopour2": (3, 3), "Ftip": (12, 4), "Fshake": (14, 4), "Famount": (12, 2)}
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
        ("da_spill2", "Famount", 1, (-0.1,5)),
    ]
    
    node_states_dim_pair = [
        ["n0", [("size_srcmouth", 1), ("material2", 4), ("dtheta2", 1), ("shake_spd", 1), ("shake_range", 1), ("shake_angle", 1)]],
        ["n2b", [("lp_pour", 3), ]],
        ["n2c", [("skill", 1), ]],
        ["n3ti", [("da_total", 1), ("lp_flow", 2), ("flow_var", 1)]],
        ["n4ti", [("da_pour", 1), ("da_spill2", 1)]],
        ["n4tir1", [(".r", 1), ]],
        ["n4tir2", [(".r", 1), ]],
        ["n3sa", [("da_total", 1), ("lp_flow", 2), ("flow_var", 1)]],
        ["n4sa", [("da_pour", 1), ("da_spill2", 1)]],
        ["n4sar1", [(".r", 1), ]],
        ["n4sar2", [(".r", 1), ]],
    ]
    
    sh, esh = get_true_and_bestpolicy_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate=False)
    suff_annotation = [
        "<b>policy's estimation</b><br />"
        "　lp_pour xz: ({:.3f}+/-{:.3f},{:.3f}+/-{:.3f})<br />".format(esh["n2b"]["lp_pour_0"][MEAN][i], esh["n2b"]["lp_pour_0"][SIGMA][i], esh["n2b"]["lp_pour_2"][MEAN][i], esh["n2b"]["lp_pour_2"][SIGMA][i])+\
        ("　<b>[tip]</b><br />" if esh["n2c"]["skill"][MEAN][i] == 0 else "　[tip]<br />") +\
        "　　da_total: {:.3f}+/-{:.3f}<br />".format(esh["n3ti"]["da_total"][MEAN][i], esh["n3ti"]["da_total"][SIGMA][i])+\
        "　　lp_flow xy: ({:.3f}+/-{:.3f},{:.3f}+/-{:.3f})<br />".format(esh["n3ti"]["lp_flow_0"][MEAN][i], esh["n3ti"]["lp_flow_0"][SIGMA][i], esh["n3ti"]["lp_flow_1"][MEAN][i], esh["n3ti"]["lp_flow_1"][SIGMA][i])+\
        "　　flow_var: {:.3f}+/-{:.3f}<br />".format(esh["n3ti"]["flow_var"][MEAN][i], esh["n3ti"]["flow_var"][SIGMA][i])+\
        "　　da_pour: {:.3f}+/-{:.3f}<br />".format(esh["n4ti"]["da_pour"][MEAN][i], esh["n4ti"]["da_pour"][SIGMA][i])+\
        "　　da_spill2: {:.3f}+/-{:.3f}<br />".format(esh["n4ti"]["da_spill2"][MEAN][i], esh["n4ti"]["da_spill2"][SIGMA][i])+\
        "　　Rdapour: {:.3f}+/-{:.3f}<br />".format(esh["n4tir1"][".r"][MEAN][i], esh["n4tir1"][".r"][SIGMA][i])+\
        "　　Rdaspill: {:.3f}+/-{:.3f}<br />".format(esh["n4tir2"][".r"][MEAN][i], esh["n4tir2"][".r"][SIGMA][i])+\
        ("　<b>[shake]</b><br />" if esh["n2c"]["skill"][MEAN][i] == 1 else "　[shake]<br />") +\
        "　　da_total: {:.3f}+/-{:.3f}<br />".format(esh["n3sa"]["da_total"][MEAN][i], esh["n3sa"]["da_total"][SIGMA][i])+\
        "　　lp_flow xy: ({:.3f}+/-{:.3f},{:.3f}+/-{:.3f})<br />".format(esh["n3sa"]["lp_flow_0"][MEAN][i], esh["n3sa"]["lp_flow_0"][SIGMA][i], esh["n3sa"]["lp_flow_1"][MEAN][i], esh["n3sa"]["lp_flow_1"][SIGMA][i])+\
        "　　flow_var: {:.3f}+/-{:.3f}<br />".format(esh["n3sa"]["flow_var"][MEAN][i], esh["n3sa"]["flow_var"][SIGMA][i])+\
        "　　da_pour: {:.3f}+/-{:.3f}<br />".format(esh["n4sa"]["da_pour"][MEAN][i], esh["n4sa"]["da_pour"][SIGMA][i])+\
        "　　da_spill2: {:.3f}+/-{:.3f}<br />".format(esh["n4sa"]["da_spill2"][MEAN][i], esh["n4sa"]["da_spill2"][SIGMA][i])+\
        "　　Rdapour: {:.3f}+/-{:.3f}<br />".format(esh["n4sar1"][".r"][MEAN][i], esh["n4sar1"][".r"][SIGMA][i])+\
        "　　Rdaspill: {:.3f}+/-{:.3f}<br />".format(esh["n4sar2"][".r"][MEAN][i], esh["n4sar2"][".r"][SIGMA][i])
    for i in range(len(esh["n2b"]["lp_pour_0"][MEAN]))]
    
    go_layout = {
        'height': 800*2*len(vis_state_dynamics_outdim_lim_pair),
        'width': 1800,
        'margin': dict(t=30, b=0),
        'legend_tracegroupgap': 800-30,
        'hoverdistance': 5,
    }
    
    transition_plot(td, log_name_list, dynamics_iodim_pair, vis_state_dynamics_outdim_lim_pair, go_layout, suff_annotation, save_dir, file_name_pref)