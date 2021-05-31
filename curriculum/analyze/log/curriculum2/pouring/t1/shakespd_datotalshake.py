from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from ..util import *
from ...tasks_domain.util import Rmodel
from ...tasks_domain.flow_ctrl import task_domain as td


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5_5_5"
    save_dir = PICTURE_DIR + model_path.replace("/","_") + "/"
    file_name_pref = "zoom2_"
    
    model_name = "Fshake"
    xs_value = {
        "gh_abs": [0.25],
        "lp_pour_x": [0.],
        "lp_pour_y": [0.],
        "lp_pour_z": [0.],
        "da_trg": [0.3],
        "size_srcmouth": [0.055],
        "material2": [0.1, 0.2, 0.0, 0.1],
        "dtheta1": [1.4e-2],
        "shake_spd": [0.6],
        "shake_range": [0.04],
        "shake_angle": [0.7],  
    }
    input_features = ["gh_abs","lp_pour_x","lp_pour_y","lp_pour_z","da_trg","size_srcmouth","material2","dtheta1",'shake_spd','shake_range','shake_angle']
    X = {"feature": "size_srcmouth", "values": np.linspace(0.02,0.08,40)}
    Y = {"feature": "shake_spd", "values": np.linspace(0.4,1.3,40)}
    z = {"feature": "da_total_shake", "output_dim": 0, "range": {MEAN: [0.45,0.455], SIGMA: [0.078,0.085]}}
    reward_function = {
        "name": "Rdatotal",
        "model": Rmodel("Fdatotal_gentle"),
        "input_features": ["da_trg","da_total_shake"],
        "format_mean": lambda pred: [pred.Y[0]],
        "format_var": lambda pred: [pred.Var[0,0]],
        "range": {MEAN: [-0.62,-0.55], SIGMA: [0.14,0.16]},
    }
    
    save_sh_dir = "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1"
    node_states_dim_pair = [
        ["n0", [("size_srcmouth",1),("material2",4),("dtheta2",1)]],
        ["n2b", [("lp_pour",3),]],
        ["n3sa", [("da_total",1),]],
    ]
    sh, esh = get_true_and_bestpolicy_est_state_histories(save_sh_dir, None, node_states_dim_pair, recreate=False)
    df = pd.DataFrame({
        "lp_pour_x": sh["n2b"]["lp_pour_0"][MEAN],
        "lp_pour_z": sh["n2b"]["lp_pour_2"][MEAN],
        # "da_total_tip": sh["n3ti"]["da_total"][MEAN],
        "da_total_shake": sh["n3sa"]["da_total"][MEAN],
        "size_srcmouth": sh["n0"]["size_srcmouth"][MEAN],
        "nobounce": [True if m2 == 0.0 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        # "ketchup": [True if m2 == 0.25 else None for m2 in sh["n0"]["material2_2"][MEAN]],
        # "dtheta2": sh["n0"]["dtheta2"][MEAN],
        "shake_spd": sh["n0"]["shake_spd"][MEAN],
        "shake_range": sh["n0"]["shake_range"][MEAN],
        "shake_angle": sh["n0"]["shake_angle"][MEAN],
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