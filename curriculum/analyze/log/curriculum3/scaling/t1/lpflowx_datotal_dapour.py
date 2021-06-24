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
    model_path = "curriculum3/scaling/full_scratch/t1/second200"
    save_sh_dir = "curriculum3/scaling/full_scratch/t1"
    save_dir = PICTURE_DIR + save_sh_dir.replace("/","_") + "/"
    file_name_pref = "ketchup_datotalstd02_"
    model_name = "Famount"
    model = None
    # with open(ROOT_PATH+"test/mms4"+"/{}_{}.pkl".format(model_name,file_name_pref), "rb") as f:
    #     model = pickle.load(f)
    #     model_path = "relearned model"
    
    xs_value = {
        'lp_pour_x': [-0.11],
        'lp_pour_y': [0.0],
        'lp_pour_z': [0.142],
        "da_trg": [0.3],
        "material2": KETCHUP,
        "da_total": [0.3],
        "lp_flow_x": [-0.022],
        "lp_flow_y": [0.0],
        "flow_var": [0.206],
    }
    input_features = ["lp_pour_x","lp_pour_y","lp_pour_z","da_trg","material2","da_total","lp_flow_x","lp_flow_y","flow_var"]
    input_vars = [0.,0.,0.,0.,0.,0.,0.,0.,0.2**2,0.,0.,0.]
    # input_vars = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    X = {"feature": "lp_flow_x", "values": np.linspace(-0.4,0.6,100)}
    Y = {"feature": "da_total", "values": np.linspace(-0.2,0.6,100)}
    z = {"feature": "da_pour", "output_dim": 0, "range": {MEAN: [-0.05,0.6], SIGMA: [-0.05,0.1]}}
    reward_function = {
        "name": "Rdapour",
        "model": Rmodel("Fdapour_gentle"),
        "input_features": ["da_trg","da_pour"],
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
        "da_total": operate_list([sh["n3ti1"]["da_total"][MEAN], sh["n3sa1"]["da_total"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        "lp_flow_x": operate_list([sh["n3ti2"]["lp_flow_0"][MEAN], sh["n3sa2"]["lp_flow_0"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        # "lp_flow_y": operate_list([sh["n3ti2"]["lp_flow_1"][MEAN], sh["n3sa2"]["lp_flow_1"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        "flow_var": operate_list([sh["n3ti2"]["flow_var"][MEAN], sh["n3sa2"]["flow_var"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        "da_pour": operate_list([sh["n4ti"]["da_pour"][MEAN], sh["n4sa"]["da_pour"][MEAN]], deal_with_None=DEAL_AS_ZERO),
        # "nobounce": [True if m2 == 0.0 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "ketchup": [True if m2 == 0.25 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "size_srcmouth": sh["n0"]["size_srcmouth"][MEAN],
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
        "skill": sh["n2c"]["skill"][MEAN],
        "episode": np.arange(0,len(sh["n0"]["dtheta2"][MEAN])),
        "comment": [""]*len(sh["n0"]["size_srcmouth"][MEAN]),
    })
    df.dropna(inplace=True)
    # df["comment"][19] = "<br />　ソース位置が高く, レシーバー奥に溢れ."

    scatter_condition_title_pair = [
        ("full scatter", [True]*len(df)),
        ("Tip", df["skill"]==0),
        ("Shake", df["skill"]==1),
        # ("scatter c1 -0.2<lp_pour_x<0", ((-0.2<df["lp_pour_x"])&(df["lp_pour_x"]<0.0))),
        # ("scatter c2 0.2<lp_pour_z<0.3", ((0.2<df["lp_pour_z"])&(df["lp_pour_z"]<0.3))),
        # ("scatter c1&c2", ((-0.2<df["lp_pour_x"])&(df["lp_pour_x"]<0.0) & (0.2<df["lp_pour_z"])&(df["lp_pour_z"]<0.3))),
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
        
    plot_dynamics_heatmap(td, model_path, save_dir, file_name_pref, model_name, xs_value, input_features, input_vars, X, Y, z, reward_function, scatter_obj_list=scatter_obj_list, updatemenu=updatemenu, model=model, is_prev_model=True)
    