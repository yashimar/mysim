from core_tool import *
from util import *
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
    
PICTURE_DIR = "/home/yashima/Pictures/"


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
        "curriculum/pouring3/full_scratch/curriculum_test/t1/first150",
    ]
    save_sh_dir = "curriculum/pouring3/full_scratch/curriculum_test/t1/first150"
    save_img_dir = PICTURE_DIR + save_sh_dir.replace("/","_") + "/"
    
    node_states_dim_pair = [
        ["n0", [("size_srcmouth",1),("material2",4),("dtheta2",1)]],
        ["n2b", [("lp_pour",3),]],
        ["n3ti", [("da_total",1),]],
    ]

    sh, esh = get_true_and_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate=False)
    df = pd.DataFrame({
        "lp_pour_x": sh["n2b"]["lp_pour_0"][MEAN],
        "lp_pour_z": sh["n2b"]["lp_pour_2"][MEAN],
        "da_total_tip": sh["n3ti"]["da_total"][MEAN],
        "size_srcmouth": sh["n0"]["size_srcmouth"][MEAN],
        "nobounce": [True if m2 == 0.0 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        # "ketchup": [True if m2 == 0.25 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
    })
    df.dropna(inplace=True)
    
    # fig = px.scatter_3d(df, x="lp_pour_x", y="lp_pour_z", z="da_total_tip")
    # fig.show()
    
    plot_and_save_df_scatter(df,[
        ("lp_pour_x", "lp_pour_z"),
        ("lp_pour_x", "da_total_tip"),
        ("lp_pour_z", "da_total_tip"),
        ("size_srcmouth", "da_total_tip"),
        ("dtheta2", "da_total_tip"),
    ], save_img_dir)