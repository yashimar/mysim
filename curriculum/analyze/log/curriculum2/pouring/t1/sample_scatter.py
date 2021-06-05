# coding: UTF-8
from core_tool import *
from .....util import *
import pandas as pd


def Help():
    pass


def Run(ct, *args):
    log_name_list = [
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
        # "da_total_tip": sh["n3ti"]["da_total"][MEAN],
        "da_total": operate_list([sh["n3ti"]["da_total"][MEAN], sh["n3sa"]["da_total"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        # "lp_flow_x_tip": sh["n3ti"]["lp_flow_0"][MEAN],
        "lp_flow_x": operate_list([sh["n3ti"]["lp_flow_0"][MEAN], sh["n3sa"]["lp_flow_0"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        # "flow_var_tip": sh["n3ti"]["flow_var"][MEAN],
        "flow_var": operate_list([sh["n3ti"]["flow_var"][MEAN], sh["n3sa"]["flow_var"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        # "da_pour_tip": sh["n4ti"]["da_pour"][MEAN],
        "da_pour": operate_list([sh["n4ti"]["da_pour"][MEAN], sh["n4sa"]["da_pour"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        # "da_spill_tip": sh["n4ti"]["da_spill2"][MEAN],
        "da_spill": operate_list([sh["n4ti"]["da_spill2"][MEAN], sh["n4sa"]["da_spill2"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        # "da_total_shake": sh["n3sa"]["da_total"][MEAN],
        "size_srcmouth": sh["n0"]["size_srcmouth"][MEAN],
        "nobounce": [True if m2 == 0.0 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "ketchup": [True if m2 == 0.25 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
        "shake_spd": sh["n0"]["shake_spd"][MEAN],
        "shake_range": sh["n0"]["shake_range"][MEAN],
        "shake_angle": sh["n0"]["shake_angle"][MEAN],
        "skill": sh["n2c"]["skill"][MEAN],
        "episode": np.arange(0, len(sh["n0"]["size_srcmouth"][MEAN])),
        # "Rdapour_Rdaspill": operate_list([sh["n4tir1"][".r"][MEAN], sh["n4tir2"][".r"][MEAN]], deal_with_None=FORCE_TO_NONE),
        "Rdapour_Rdaspill": operate_list([sh["n4tir1"][".r"][MEAN], sh["n4tir2"][".r"][MEAN], sh["n4sar1"][".r"][MEAN], sh["n4sar2"][".r"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        "comment": [""]*len(sh["n0"]["size_srcmouth"][MEAN]),
    })
    for c in df.columns:
        if "shake" in c:
            df[c][df["skill"]!=1] = None
        elif "dtheta2" in c:
            df[c][df["skill"]!=0] = None
    # df.dropna(inplace=True)
    df["comment"][19] = "<br />　ソース位置が高く, レシーバー奥に溢れ."
    df["comment"][27] = "<br />　'flow_out'遷移後, dtheta2が大きくすぐに最大角に到達して終了."\
                        + "<br />　目標量出したことによって終了しておらず, kickback中に多量流れ出ており, da_total_tipが目標量に近いのは偶然."
    df["comment"][50] = "<br />　kickbackの反動で, レシーバー手前に溢れ."
    df["comment"][78] = "<br />　ソース位置がやや高く奥まっており, レシーバー奥に溢れ."
    df["comment"][79] = "<br />　レシーバーの右上で注いでしまい, すべて溢れる."\
                        + "<br />　dtheta2が大きく最大角に到達して終了."\
                        + "<br />　口径が大きく'kickback'時に多く流れ出る."
    df["comment"][87] = "<br />　レシーバーより手前過ぎて溢れ."
    df["comment"][92] = "<br />　'flow_out'遷移後, 流れ出るまでの待機時間が上限に達し, 最大角に到達することなく終了."\
                        + "<br />　遷移後すぐに少し流れ出たが, その後流れ出なくなった."\
                        + "<br />　dtheta2下限値."\
                        + "<br />　dtheta2を大きくするか待機時間を伸ばすと注げる."
    df["comment"][176] = "<br />　'flow_out'遷移後, dtheta2が大きいために, 十分な量が出ることなく最大角に到達して終了."\
                        + "<br />　dtheta2が小さければ, 十分な量を流し出せたはずの動作."
    df["comment"][196] = "<br />　レシーバーより手前過ぎて溢れ."
    df["comment"][199] = "<br />　悪い局所解に最適化."\
                        + "<br />　[tip] Rdapour: -2, Rdaspill2: -14"\
                        + "<br />　[shake] Rdapour: -1, Rdaspill2: -6"
    df["comment"][239] = "<br />　'flow_out'遷移後, da_totalが目標量に到達したため終了."\
                        + "<br />　dtheta2が大きいため, 傾きが大きい状態で最初の流出が始まり, 一気に流出したため目標量を大きく超えた."
    df["comment"][269] = "<br />　'flow_var','lp_flow'がエピソード中更新されなかったためゼロ."\
                        + "<br />　球体が'レシーバーの中'かつ'地面以外と接触'のとき'flow状態'とみなされない."\
                        +"<br />　'flow状態'の球体が床面から10cm以下のときに'flow_var','lp_flow'が計算される.'"\
                        +"<br />　床面から10cmはぎりぎりレシーバー底に触れない高さ.'"\
                        +"<br />　'flow状態'かつ床面から10cm以下になる球が発生しなかったため'flow_var','lp_flow'が初期値0のまま."\
                        +"<br />　'flow状態'かつ床面から12cm程度であれば発生していたので要調整."\
                        +"<br />　ay_trick/sm4.pyのL114."\
                        +"<br />　ay_sim/ode_grpour_sim.cppのL894, L873."
    vis_df_title_pair = [
        (df, "full data"), 
        (df[df["nobounce"]==True], "all skill, nobounce only"),
        (df[df["ketchup"]==True], "all skill, ketchup only"),
        (df[(df["skill"]==0)&(df["nobounce"]==True)], "tip, nobounce only"),
        (df[(df["skill"]==1)&(df["nobounce"]==True)], "shake, nobounce only"),
        (df[(df["skill"]==0)&(df["ketchup"]==True)], "tip, ketchup only"),
        (df[(df["skill"]==1)&(df["ketchup"]==True)], "shake, ketchup only"),
    ]
    
    xy_limit_pairs = [
        ("episode", "Rdapour_Rdaspill", [-10, len(df)+10], [-40,0.5]),
        ("lp_pour_x", "lp_pour_z", [-0.5, 0.7], [-0.2, 0.6]),
        ("lp_pour_x", "da_total", [-0.5, 0.7], [-0.1, 0.6]),
        ("lp_pour_z", "da_total", [-0.2, 0.6], [-0.1, 0.6]),
        ("dtheta2", "da_total", [0., 0.025], [-0.1, 0.6]),
        ("size_srcmouth", "da_total", [0.01, 0.10], [-0.1, 0.6]),
        ("lp_pour_x", "da_pour", [-0.5, 0.7], [-0.1, 0.6]),
        ("lp_pour_z", "da_pour", [-0.2, 0.6], [-0.1, 0.6]),
        ("dtheta2", "da_pour", [0., 0.025], [-0.1, 0.6]),
        ("size_srcmouth", "da_pour", [0.01, 0.10], [-0.1, 0.6]),
        ("lp_pour_x", "da_spill", [-0.5, 0.7], [-0.1, 10]),
        ("lp_pour_x", "da_spill", [-0.5, 0.7], [-0.1, 1]),
        ("lp_pour_z", "da_spill", [-0.2, 0.6], [-0.1, 10]),
        ("lp_pour_z", "da_spill", [-0.2, 0.6], [-0.1, 1]),
        ("dtheta2", "da_spill", [0., 0.025], [-0.1, 10]),
        ("dtheta2", "da_spill", [0., 0.025], [-0.1, 1]),
        ("size_srcmouth", "da_spill", [0.01, 0.10], [-0.1, 10]),
        ("size_srcmouth", "da_spill", [0.01, 0.10], [-0.1, 1]),
        ("lp_flow_x", "flow_var", [-0.5, 0.7], [-0.1, 0.6]),
        ("lp_pour_x", "lp_flow_x", [-0.5, 0.7], [-0.5, 0.7]),
        ("lp_flow_x", "da_pour", [-0.5, 0.7], [-0.1, 0.6]),
        ("lp_flow_x", "da_spill", [-0.5, 0.7], [-0.1, 1]),
        ("flow_var", "da_pour", [-0.1, 0.6], [-0.1, 0.6]),
        ("flow_var", "da_spill", [-0.1, 0.6], [-0.1, 1]),
        ("shake_spd", "da_total", [0.4, 1.3], [-0.1, 0.6]),
        ("shake_range", "da_total", [0.04, 0.13], [-0.1, 0.6]),
        ("shake_angle", "da_total", [-0.6*math.pi, 0.6*math.pi], [-0.1, 0.6]),
    ]
         
    def updatemenu(fig):
        buttons = [{
            'label': vis_title,
            'method': "update",
            'args':[
                {'visible': [True if j%len(vis_df_title_pair) == i else False for j in range(len(vis_df_title_pair)*len(xy_limit_pairs))]},
                # {'title': ["[{}] {} / {}".format(title,x,y) for x,y,_,_ in xy_limit_pairs for _,title in vis_df_title_pair]}
            ]
        } for i, (_, vis_title) in enumerate(vis_df_title_pair)]
        updatemenus = [{
            "type": "dropdown",
            "buttons": buttons,
            "active": 0,
            # "x": 1.05,
            # "xanchor": "left",
            # "y": 1,
            # "yanchor": "top",
        }]
        fig['layout']['updatemenus'] = updatemenus
    # updatemenu = None

    go_layout = {
        'height': 14000,
        'width': 900,
        'margin': dict(t=150, b=20),
        'hoverdistance': 5,
    }
    color = lambda df: [1. if ((-0.114<lpx<-0.107) or (0.139<lpz<0.145) or ((-0.1<lpx<0) and (0.25<lpz<0.32)) or ((0<lpx<0.08) and (0.315<lpz<0.32)) or ((0.16<lpx<0.2) and (0.32<lpz<0.35))) else 0. for lpx,lpz in zip(df["lp_pour_x"],df["lp_pour_z"])]
    
    plot_and_save_df_scatter(vis_df_title_pair, xy_limit_pairs, save_img_dir, concat_title="sample_scatter"+"_c_lppourx", color=color, go_layout=go_layout, updatemenu=updatemenu)
    
