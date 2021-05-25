from core_tool import *
from util import *
import pandas as pd


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
    save_sh_dir = "curriculum2/pouring/full_scratch/curriculum_test/t1"
    save_img_dir = PICTURE_DIR + save_sh_dir.replace("/", "_") + "/"

    node_states_dim_pair = [
        ["n0", [("size_srcmouth", 1), ("material2", 4), ("dtheta2", 1), ("shake_spd", 1), ("shake_range", 1), ("shake_angle", 1)]],
        ["n2b", [("lp_pour", 3), ]],
        ["n3ti", [("da_total", 1), ]],
        ["n4tir1", [(".r", 1), ]],
        ["n4tir2", [(".r", 1), ]],
        ["n3sa", [("da_total", 1), ]],
        ["n4sar1", [(".r", 1), ]],
        ["n4sar2", [(".r", 1), ]],
    ]

    sh, esh = get_true_and_bestpolicy_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate=False)
    df = pd.DataFrame({
        "lp_pour_x": sh["n2b"]["lp_pour_0"][MEAN],
        "lp_pour_z": sh["n2b"]["lp_pour_2"][MEAN],
        "da_total_tip": sh["n3ti"]["da_total"][MEAN],
        # "da_total_shake": sh["n3sa"]["da_total"][MEAN],
        "size_srcmouth": sh["n0"]["size_srcmouth"][MEAN],
        "nobounce": [True if m2 == 0.0 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        # "ketchup": [True if m2 == 0.25 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
        # "shake_spd": sh["n0"]["shake_spd"][MEAN],
        # "shake_range": sh["n0"]["shake_range"][MEAN],
        # "shake_angle": sh["n0"]["shake_angle"][MEAN],
        "episode": np.arange(0, len(sh["n0"]["dtheta2"][MEAN])),
        "Rdapour_Rdaspill": operate_list([sh["n4tir1"][".r"][MEAN], sh["n4tir2"][".r"][MEAN]], deal_with_None=FORCE_TO_NONE),
    })
    df.dropna(inplace=True)

    # fig = px.scatter_3d(df, x="lp_pour_x", y="lp_pour_z", z="da_total_tip")
    # fig.show()

    go_layout = {
        'height': 4000,
        'width': 900,
        'margin': dict(t=20, b=150),
        'hoverdistance': 5,
    }
    plot_and_save_df_scatter(df, [
        # ("episode", "Rdapour_Rdaspill", None, [-20,0.5]),
        ("lp_pour_x", "lp_pour_z", [-0.5, 0.7], [-0.2, 0.6]),
        ("lp_pour_x", "da_total_tip", [-0.5, 0.7], [-0.1, 0.6]),
        ("lp_pour_z", "da_total_tip", [-0.2, 0.6], [-0.1, 0.6]),
        ("dtheta2", "da_total_tip", [0., 0.025], [-0.1, 0.6]),
        ("size_srcmouth", "da_total_tip", [0.01, 0.10], [-0.1, 0.6]),
        # ("shake_spd", "da_total_shake", [0.4, 1.3], [-0.1, 0.6]),
        # ("shake_range", "da_total_shake", [0.04, 0.13], [-0.1, 0.6]),
        # ("shake_angle", "da_total_shake", [-0.6*math.pi, 0.6*math.pi], [-0.1, 0.6]),
    ], save_img_dir, concat_title="tip_nobounce", go_layout=go_layout)
    
