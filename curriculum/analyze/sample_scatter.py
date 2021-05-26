# coding: UTF-8
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
    df = pd.DataFrame({
        "lp_pour_x": sh["n2b"]["lp_pour_0"][MEAN],
        "lp_pour_z": sh["n2b"]["lp_pour_2"][MEAN],
        "da_total_tip": sh["n3ti"]["da_total"][MEAN],
        "lp_flow_x_tip": sh["n3ti"]["lp_flow_0"][MEAN],
        "flow_var_tip": sh["n3ti"]["flow_var"][MEAN],
        "da_pour_tip": sh["n4ti"]["da_pour"][MEAN],
        "da_spill_tip": sh["n4ti"]["da_spill2"][MEAN],
        # "da_total_shake": sh["n3sa"]["da_total"][MEAN],
        "size_srcmouth": sh["n0"]["size_srcmouth"][MEAN],
        "nobounce": [True if m2 == 0.0 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        # "ketchup": [True if m2 == 0.25 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
        # "shake_spd": sh["n0"]["shake_spd"][MEAN],
        # "shake_range": sh["n0"]["shake_range"][MEAN],
        # "shake_angle": sh["n0"]["shake_angle"][MEAN],
        "episode": np.arange(0, len(sh["n0"]["size_srcmouth"][MEAN])),
        "Rdapour_Rdaspill": operate_list([sh["n4tir1"][".r"][MEAN], sh["n4tir2"][".r"][MEAN]], deal_with_None=FORCE_TO_NONE),
        "comment": [""]*len(sh["n0"]["size_srcmouth"][MEAN]),
    })
    # df.dropna(inplace=True)
    df["comment"][19] = "<br />　ソース位置が高く, レシーバー奥に溢れ."
    df["comment"][27] = "<br />　'flow_out'遷移後, dtheta2が大きくすぐに最大角に到達して終了."\
                        + "<br />　目標量出したことによって終了しておらず, kickback中に多量流れ出ており, da_total_tipが目標量に近いのは偶然."
    df["comment"][50] = "<br />　kickbackの反動で, レシーバー手前に溢れ."
    df["comment"][78] = "<br />　ソース位置がやや高く奥まっており, レシーバー奥に溢れ."
    df["comment"][87] = "<br />　レシーバーより手前過ぎて溢れ."
    df["comment"][92] = "<br />　'flow_out'遷移後, 流れ出るまでの待機時間が上限に達し, 最大角に到達することなく終了."\
                        + "<br />　遷移後すぐに少し流れ出たが, その後流れ出なくなった."\
                        + "<br />　稀な現象."
    df["comment"][176] = "<br />　'flow_out'遷移後, dtheta2が大きいために, 十分な量が出ることなく最大角に到達して終了."\
                        + "<br />　dtheta2が小さければ, 十分な量を流し出せたはずの動作."
    df["comment"][196] = "<br />　レシーバーより手前過ぎて溢れ."
    df["comment"][239] = "<br />　'flow_out'遷移後, da_totalが目標量に到達したため終了."\
                        + "<br />　dtheta2が大きいため, 傾きが大きい状態で最初の流出が始まり, 一気に流出したため目標量を大きく超えた."

    go_layout = {
        'height': 12000,
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
        ("lp_pour_x", "da_pour_tip", [-0.5, 0.7], [-0.1, 0.6]),
        ("lp_pour_z", "da_pour_tip", [-0.2, 0.6], [-0.1, 0.6]),
        ("dtheta2", "da_pour_tip", [0., 0.025], [-0.1, 0.6]),
        ("size_srcmouth", "da_pour_tip", [0.01, 0.10], [-0.1, 0.6]),
        ("lp_pour_x", "da_spill_tip", [-0.5, 0.7], [-0.1, 10]),
        ("lp_pour_x", "da_spill_tip", [-0.5, 0.7], [-0.1, 1]),
        ("lp_pour_z", "da_spill_tip", [-0.2, 0.6], [-0.1, 10]),
        ("lp_pour_z", "da_spill_tip", [-0.2, 0.6], [-0.1, 1]),
        ("dtheta2", "da_spill_tip", [0., 0.025], [-0.1, 10]),
        ("dtheta2", "da_spill_tip", [0., 0.025], [-0.1, 1]),
        ("size_srcmouth", "da_spill_tip", [0.01, 0.10], [-0.1, 10]),
        ("size_srcmouth", "da_spill_tip", [0.01, 0.10], [-0.1, 1]),
        ("lp_flow_x_tip", "flow_var_tip", [-0.5, 0.7], [-0.1, 0.6]),
        ("lp_flow_x_tip", "da_pour_tip", [-0.5, 0.7], [-0.1, 0.6]),
        ("lp_flow_x_tip", "da_spill_tip", [-0.5, 0.7], [-0.1, 1]),
        ("flow_var_tip", "da_pour_tip", [-0.1, 0.6], [-0.1, 0.6]),
        ("flow_var_tip", "da_spill_tip", [-0.1, 0.6], [-0.1, 1]),
        # ("shake_spd", "da_total_shake", [0.4, 1.3], [-0.1, 0.6]),
        # ("shake_range", "da_total_shake", [0.04, 0.13], [-0.1, 0.6]),
        # ("shake_angle", "da_total_shake", [-0.6*math.pi, 0.6*math.pi], [-0.1, 0.6]),
    ], save_img_dir, concat_title="tip_ketchup", go_layout=go_layout)
    
