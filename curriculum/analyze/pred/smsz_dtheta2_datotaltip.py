from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from util import *
from ..tasks_domain.util import Rmodel
from ..tasks_domain.flow_ctrl import task_domain as td


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5_5_5"
    save_dir = PICTURE_DIR + model_path.replace("/","_") + "/"
    file_name_pref = "zoom_"
    
    model_name = "Ftip"
    xs_value = {
        "gh_abs": [0.25],
        "lp_pour_x": [-0.1],
        "lp_pour_y": [0.],
        "lp_pour_z": [0.25],
        "da_trg": [0.3],
        "size_srcmouth": [0.055],
        "material2": [0.1, 0.2, 0.0, 0.1],
        "dtheta1": [1.4e-2],
        "dtheta2": [0.002],
    }
    input_features = ["gh_abs","lp_pour_x","lp_pour_y","lp_pour_z","da_trg","size_srcmouth","material2","dtheta1","dtheta2"]
    X = {"feature": "size_srcmouth", "values": np.linspace(0.02,0.09,40)}
    Y = {"feature": "dtheta2", "values": np.linspace(0.0,0.025,40)}
    z = {"feature": "da_total_tip", "output_dim": 0, "range": {MEAN: [0.28,0.34], SIGMA: [0.11,0.14]}}
    reward_function = {
        "name": "Rdatotal",
        "model": Rmodel("Fdatotal_gentle"),
        "input_features": ["da_trg","da_total_tip"],
        "format_mean": lambda pred: [pred.Y[0]],
        "format_var": lambda pred: [pred.Var[0,0]],
        "range": {MEAN: [-1.2,-0.6], SIGMA: [0.2,0.38]}
    }
    
    save_sh_dir = "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1"
    node_states_dim_pair = [
        ["n0", [("size_srcmouth",1),("material2",4),("dtheta2",1)]],
        ["n2b", [("lp_pour",3),]],
        ["n3ti", [("da_total",1),]],
    ]
    sh, esh = get_true_and_est_state_histories(save_sh_dir, None, node_states_dim_pair, recreate=False)
    df = pd.DataFrame({
        "size_srcmouth": sh["n0"]["size_srcmouth"][MEAN],
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
        "lp_pour_x": sh["n2b"]["lp_pour_0"][MEAN],
        "lp_pour_z": sh["n2b"]["lp_pour_2"][MEAN],
        "da_total_tip": sh["n3ti"]["da_total"][MEAN],
        "nobounce": [True if m2 == 0.0 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        # "ketchup": [True if m2 == 0.25 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        "dtheta2": sh["n0"]["dtheta2"][MEAN],
        "episode": np.arange(0,len(sh["n0"]["dtheta2"][MEAN])),
    })
    df.dropna(inplace=True)
    scatter_obj = go.Scatter(
        x=df[X["feature"]], y=df[Y["feature"]], 
        mode='markers', 
        # marker_color="blue",
        opacity = 0.5,
        hoverinfo='text',
        text=["true {}: {}<br />{}: {}<br />{}: {}<br />{}: {}".format(z["feature"], v_z, X["feature"], v_x, Y["feature"], v_y, "ep", v_ep) for v_z, v_x, v_y, v_ep in zip(df[z["feature"]], df[X["feature"]], df[Y["feature"]], df["episode"])],
        showlegend=False,
        marker = dict(
            size = 10,
            color = df[z["feature"]].values,
            colorscale="Viridis",
            cmin = 0,
            cmax = 0.55,
        ),
    )
    
    plot_dynamics_heatmap(td, model_path, save_dir, file_name_pref, model_name, xs_value, input_features, X, Y, z, reward_function, scatter_obj=scatter_obj)
