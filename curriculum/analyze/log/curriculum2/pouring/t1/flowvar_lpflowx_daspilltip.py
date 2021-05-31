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
    save_dir = PICTURE_DIR + model_path.replace("/","_") + "/"
    file_name_pref = "ep299_"
    model_name = "Famount"
    model = None
    # with open(ROOT_PATH+"test/mms4"+"/{}_{}.pkl".format(model_name,file_name_pref), "rb") as f:
    #     model = pickle.load(f)
    #     model_path = "relearned model"
    
    xs_value = {
        "lp_pour_x": [0.06],
        "lp_pour_y": [0.],
        "lp_pour_z": [0.318],
        "da_trg": [0.3],
        "material2": KETCHUP,
        "da_total": [0.4],
        "lp_flow_x": [],
        "lp_flow_y": [0.],
        "flow_var": []
    }
    input_features = ["lp_pour_x","lp_pour_y","lp_pour_z","da_trg","material2","da_total","lp_flow_x","lp_flow_y","flow_var"]
    X = {"feature": "lp_flow_x", "values": np.linspace(-0.36,0.26,40)}
    Y = {"feature": "flow_var", "values": np.linspace(-0.05,0.55,40)}
    z = {"feature": "da_spill_tip", "output_dim": 1, "range": {MEAN: [-0.05,5.0], SIGMA: [0.,0.4]}}
    reward_function = {
        "name": "Rdaspill",
        "model": Rmodel("Fdaspill"),
        "input_features": ["da_spill_tip"],
        "format_mean": lambda pred: [pred.Y[1]],
        "format_var": lambda pred: [pred.Var[1,1]],
        "range": {MEAN: [-6.,0.], SIGMA: [-0.05,1.0]}
    }
    
    save_sh_dir = "curriculum2/pouring/full_scratch/curriculum_test/t1"
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
        "da_total": sh["n3ti"]["da_total"][MEAN],
        "lp_flow_x": sh["n3ti"]["lp_flow_0"][MEAN],
        "flow_var": sh["n3ti"]["flow_var"][MEAN],
        "da_spill_tip": sh["n4ti"]["da_spill2"][MEAN],
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
    df["comment"][79] = "<br />　レシーバーの右上で注いでしまい, すべて溢れる."\
                        + "<br />　dtheta2が大きく最大角に到達して終了."\
                        + "<br />　口径が大きく'kickback'時に多く流れ出る."
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
    df["comment"][269] = "<br />　flow_varがエピソード中更新されなかったためゼロ."\
                        + "<br />　球体が'レシーバーの中'かつ'地面以外と接触'のとき'flow状態'とみなされない."\
                        +"<br />　'flow状態'の球体が床面から10cm以下のときに'flow_var'が計算される.'"\
                        +"<br />　床面から10cmはぎりぎりレシーバー底に触れない高さ.'"\
                        +"<br />　'flow状態'かつ床面から10cm以下になる球が発生しなかったため'flow_var'が初期値0のまま."\
                        +"<br />　'flow状態'かつ床面から12cm程度であれば発生していたので要調整."\
                        +"<br />　ay_trick/sm4.pyのL114."\
                        +"<br />　ay_sim/ode_grpour_sim.cppのL894, L873."
    
    scatter_condition_title_pair = [
        ("full scatter", [True]*len(df)),
        ("scatter c1\n0.35<da_total<0.45", (0.05<df["da_total"])<0.06),
        # ("scatter c2\n0.002<dtheta2<0.005", (0.002<df["dtheta2"])<0.005),
        # ("scatter c1&c2", ((0.05<df["size_srcmouth"])<0.06) & ((0.002<df["dtheta2"])<0.005)),
        ("no scatter", [False]*len(df)),
    ]
    scatter_obj_list = [
        go.Scatter(
            x=df[condition][X["feature"]], y=df[condition][Y["feature"]], 
            mode='markers', 
            # marker_color="blue",
            opacity = 0.5,
            hoverinfo='text',
            # text=["true {}: {}<br />{}: {}<br />{}: {}<br />{}: {}".format(z["feature"], v_z, X["feature"], v_x, Y["feature"], v_y, "ep", v_ep) for v_z, v_x, v_y, v_ep in zip(df[condition][z["feature"]], df[condition][X["feature"]], df[condition][Y["feature"]], df[condition]["episode"])],
            text=["".join(["{}: {}<br />".format(c, df[condition][c][i]) for c in df[condition].columns if c!="comment"])+("<b>comment</b>: {}".format(df[condition]["comment"][i]) if df[condition]["comment"][i] != "" else "") for i in df[condition].index],
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
