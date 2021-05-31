# coding: UTF-8
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .....util import *
from ......tasks_domain.util import Rmodel
from ......tasks_domain.pouring import task_domain as td


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum2/pouring/full_scratch/curriculum_test/t1/first300"
    save_sh_dir = "curriculum2/pouring/full_scratch/curriculum_test/t1"
    save_dir = PICTURE_DIR + save_sh_dir.replace("/","_") + "/"
    file_name_pref = "ketchup_"
    model_name = "Ftip"
    model = None
    # with open(ROOT_PATH+"test/mms4"+"/{}_{}.pkl".format(model_name,file_name_pref), "rb") as f:
    #     model = pickle.load(f)
    #     model_path = "relearned model"
    
    xs_value = {
        "gh_abs": [0.25],
        "lp_pour_x": [-0.1],
        "lp_pour_y": [0.],
        "lp_pour_z": [0.25],
        "da_trg": [0.3],
        "size_srcmouth": [0.055],
        "material2": KETCHUP,
        "dtheta1": [1.4e-2],
        "dtheta2": [0.002],
    }
    input_features = ["gh_abs","lp_pour_x","lp_pour_y","lp_pour_z","da_trg","size_srcmouth","material2","dtheta1","dtheta2"]
    X = {"feature": "size_srcmouth", "values": np.linspace(0.02,0.09,40)}
    Y = {"feature": "dtheta2", "values": np.linspace(0.0,0.025,40)}
    z = {"feature": "da_total_tip", "output_dim": 0, "range": {MEAN: [-0.05,0.8], SIGMA: [-0.05,0.1]}}
    reward_function = {
        "name": "Rdatotal",
        "model": Rmodel("Fdatotal_gentle"),
        "input_features": ["da_trg","da_total_tip"],
        "format_mean": lambda pred: [pred.Y[0]],
        "format_var": lambda pred: [pred.Var[0,0]],
        "range": {MEAN: [-3.,0.], SIGMA: [-0.05,2.0]}
    }
    
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
    sh, esh = get_true_and_bestpolicy_est_state_histories(save_sh_dir, None, node_states_dim_pair, recreate=False)
    df = pd.DataFrame({
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
        "lp_pour_x": sh["n2b"]["lp_pour_0"][MEAN],
        "lp_pour_z": sh["n2b"]["lp_pour_2"][MEAN],
        "da_total_tip": sh["n3ti"]["da_total"][MEAN],
        # "nobounce": [True if m2 == 0.0 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "ketchup": [True if m2 == 0.25 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "size_srcmouth": sh["n0"]["size_srcmouth"][MEAN],
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
        "episode": np.arange(0,len(sh["n0"]["dtheta2"][MEAN])),
        "comment": [""]*len(sh["n0"]["size_srcmouth"][MEAN]),
    })
    df.dropna(inplace=True)
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
    df["comment"][199] = "<br />　悪い局所解に最適化."\
                        + "<br />　[tip] Rdapour: -2, Rdaspill2: -14"\
                        + "<br />　[shake] Rdapour: -1, Rdaspill2: -6"
    df["comment"][239] = "<br />　'flow_out'遷移後, da_totalが目標量に到達したため終了."\
                        + "<br />　dtheta2が大きいため, 傾きが大きい状態で最初の流出が始まり, 一気に流出したため目標量を大きく超えた."
    scatter_condition_title_pair = [
        ("full scatter", [True]*len(df)),
        ("scatter c1 -0.2<lp_pour_x<0", ((-0.2<df["lp_pour_x"])&(df["lp_pour_x"]<0.0))),
        ("scatter c2 0.2<lp_pour_z<0.3", ((0.2<df["lp_pour_z"])&(df["lp_pour_z"]<0.3))),
        ("scatter c1&c2", ((-0.2<df["lp_pour_x"])&(df["lp_pour_x"]<0.0) & (0.2<df["lp_pour_z"])&(df["lp_pour_z"]<0.3))),
        ("no scatter", [False]*len(df)),
    ]
    print(-0.2<df["lp_pour_x"])
    scatter_obj_list = [
        go.Scatter(
            x=df[condition][X["feature"]], y=df[condition][Y["feature"]], 
            mode='markers', 
            # marker_color="blue",
            opacity = 0.5,
            hoverinfo='text',
            # text=["true {}: {}<br />{}: {}<br />{}: {}<br />{}: {}".format(z["feature"], v_z, X["feature"], v_x, Y["feature"], v_y, "ep", v_ep) for v_z, v_x, v_y, v_ep in zip(df[condition][z["feature"]], df[condition][X["feature"]], df[condition][Y["feature"]], df[condition]["episode"])],
            text=["".join(["{}: {}<br />".format(c, df[c][i]) for c in df.columns if c!="comment"])+("<b>comment</b>: {}".format(df["comment"][i]) if df["comment"][i] != "" else "") for i in df.index],
            showlegend=False,
            marker = dict(
                size = 10,
                color = df[condition][z["feature"]].values,
                colorscale="Viridis",
                cmin = 0,
                cmax = 0.55,
            ),
        )
    for _, condition in scatter_condition_title_pair]
    # scatter_obj = go.Scatter(
    #     x=df[X["feature"]], y=df[Y["feature"]], 
    #     mode='markers', 
    #     # marker_color="blue",
    #     opacity = 0.5,
    #     hoverinfo='text',
    #     text=["true {}: {}<br />{}: {}<br />{}: {}<br />{}: {}".format(z["feature"], v_z, X["feature"], v_x, Y["feature"], v_y, "ep", v_ep) for v_z, v_x, v_y, v_ep in zip(df[z["feature"]], df[X["feature"]], df[Y["feature"]], df["episode"])],
    #     showlegend=False,
    #     marker = dict(
    #         size = 10,
    #         color = df[z["feature"]].values,
    #         colorscale="Viridis",
    #         cmin = 0,
    #         cmax = 0.55,
    #     ),
    # )
    
    def updatemenu(fig):
        buttons = [{
            'label': title,
            'method': "update",
            'args':[
                    {'visible': [True if ((j%(len(scatter_condition_title_pair)+1)==0) or (j%(len(scatter_condition_title_pair)+1)==(i+1))) else False for j in range((len(scatter_condition_title_pair)+1)*4)]},
                ]
        } for i, (title, _) in enumerate(scatter_condition_title_pair)]
        updatemenus = [{
            "type": "dropdown",
            "buttons": buttons,
            "active": 0,
            "x": 0.0,
            "xanchor": 'left',
            "y": 1.1,
            "yanchor": 'top',
        }]
        fig['layout']['updatemenus'] = updatemenus
    # updatemenu = None
    
    go_layout = {
        'height': 14000,
        'width': 9000,
        'margin': dict(t=150, b=20),
        'hoverdistance': 5,
    }
        
    plot_dynamics_heatmap(td, model_path, save_dir, file_name_pref, model_name, xs_value, input_features, X, Y, z, reward_function, scatter_obj_list=scatter_obj_list, updatemenu=updatemenu, model=model)
    