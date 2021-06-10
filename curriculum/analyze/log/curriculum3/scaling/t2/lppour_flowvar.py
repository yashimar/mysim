# coding: UTF-8
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .....util import *
from ......tasks_domain.detach_datotal.util import Rmodel
from ......tasks_domain.detach_datotal.scaling import task_domain as td
AMP_DTHETA2, AMP_SMSZ, AMP_SHAKE_RANGE = td.AMP_DTHETA2, td.AMP_SMSZ, td.AMP_SHAKE_RANGE


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum3/scaling/full_scratch/t2/first100"
    save_sh_dir = "curriculum3/scaling/full_scratch/t2"
    save_dir = PICTURE_DIR + save_sh_dir.replace("/","_") + "/"
    file_name_pref = "nobounce_"
    model_name = "Ftip_flow"
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
        "size_srcmouth": [0.055*AMP_SMSZ],
        "material2": NOBOUNCE,
        "dtheta1": [1.4e-2],
        "dtheta2": [0.002*AMP_DTHETA2],
        "da_total_tip": [0.],
    }
    input_features = ["gh_abs","lp_pour_x","lp_pour_y","lp_pour_z","da_trg","size_srcmouth","material2","dtheta1","dtheta2"]
    X = {"feature": "lp_pour_x", "values": np.linspace(-0.5,0.7,100)}
    Y = {"feature": "lp_pour_z", "values": np.linspace(-0.2,0.6,100)}
    z = {"feature": "flow_var_tip", "output_dim": 2, "range": {MEAN: [0.1,0.5], SIGMA: [-0.05,0.15]}}
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
        ["n3ti1", [("da_total", 1),]],
        ["n3ti2", [("lp_flow", 2), ("flow_var", 1)]],
        ["n4ti", [("da_pour", 1), ("da_spill2", 1)]],
        ["n4tir1", [(".r", 1), ]],
        ["n4tir2", [(".r", 1), ]],
        ["n3sa1", [("da_total", 1),]],
        ["n3sa2", [("lp_flow", 2), ("flow_var", 1)]],
        ["n4sa", [("da_pour", 1), ("da_spill2", 1)]],
        ["n4sar1", [(".r", 1), ]],
        ["n4sar2", [(".r", 1), ]],
    ]
    sh, esh = get_true_and_bestpolicy_est_state_histories(save_sh_dir, [model_path], node_states_dim_pair, recreate=False)
    df = pd.DataFrame({
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
        "lp_pour_x": sh["n2b"]["lp_pour_0"][MEAN],
        "lp_pour_z": sh["n2b"]["lp_pour_2"][MEAN],
        "da_total_tip": sh["n3ti1"]["da_total"][MEAN],
        "flow_var_tip": sh["n3ti2"]["flow_var"][MEAN],
        "nobounce": [True if m2 == 0.0 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        # "ketchup": [True if m2 == 0.25 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "size_srcmouth": sh["n0"]["size_srcmouth"][MEAN],
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
        "episode": np.arange(0,len(sh["n0"]["dtheta2"][MEAN])),
        "comment": [""]*len(sh["n0"]["size_srcmouth"][MEAN]),
    })
    df.dropna(inplace=True)
    # df["comment"][19] = "<br />　ソース位置が高く, レシーバー奥に溢れ."

    scatter_condition_title_pair = [
        ("full scatter", [True]*len(df)),
        ("scatter c1\n0.05*AMP_DTHETA2<smsz<0.07*AMP_DTHETA2", ((0.05*AMP_SMSZ<df["size_srcmouth"])&(df["size_srcmouth"]<0.07*AMP_SMSZ))),
        ("scatter c2\n0.002*AMP_SMSZ<dtheta2<0.005*AMP_SMSZ", ((0.002*AMP_DTHETA2<=df["dtheta2"])&(df["dtheta2"]<0.005*AMP_DTHETA2))),
        ("scatter c1&c2", ((0.05*AMP_SMSZ<df["size_srcmouth"])&(df["size_srcmouth"]<0.07*AMP_SMSZ) & (0.002*AMP_DTHETA2<=df["dtheta2"])&(df["dtheta2"]<0.005*AMP_DTHETA2))),
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
        
    plot_dynamics_heatmap(td, model_path, save_dir, file_name_pref, model_name, xs_value, input_features, X, Y, z, reward_function, scatter_obj_list=scatter_obj_list, updatemenu=updatemenu, model=model, is_prev_model=True)
    