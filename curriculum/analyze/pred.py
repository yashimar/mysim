from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from util import *
from ..tasks_domain.util import Rmodel
from ..tasks_domain.flow_ctrl import task_domain as td
from ..tasks_domain.util import ModelManager
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go
import chainer.functions as F


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5"
    save_dir = PICTURE_DIR + model_path.replace("/","_") + "/"
    file_name_pref = ""
    
    model_name = "Ftip"
    xs_value = {
        "gh_abs": [0.25],
        "lp_pour_x": [0.],
        "lp_pour_y": [0.],
        "lp_pour_z": [0.],
        "da_trg": [0.3],
        "size_srcmouth": [0.055],
        "material2": [0.1, 0.2, 0.0, 0.1],
        "dtheta1": [1.4e-2],
        "dtheta2": [0.002],
    }
    input_features = ["gh_abs","lp_pour_x","lp_pour_y","lp_pour_z","da_trg","size_srcmouth","material2","dtheta1","dtheta2"]
    X = {"feature": "lp_pour_x", "values": np.linspace(-0.5,0.7,40)}
    Y = {"feature": "lp_pour_z", "values": np.linspace(-0.2,0.6,40)}
    z = {"feature": "da_total", "output_dim": 0, "range": {MEAN: [-0.05,0.8], SIGMA: [-0.05,0.4]}}
    reward_function = {
        "name": "Rdamount",
        "model": Rmodel("Fdatotal_gentle"),
        "input_features": ["da_trg","da_total"],
        "format_mean": lambda pred: [pred.Y[0]],
        "format_var": lambda pred: [pred.Var[0,0]],
        "range": {MEAN: [-10.,0.], SIGMA: [-0.05,2.0]}
    }

    domain = td.Domain()
    mm = ModelManager(domain, ROOT_PATH+model_path)
    model = mm.Models[model_name][2]
    inputs = []
    for v_y in Y["values"]:
        for v_x in X["values"]:
            input_tmp = []
            xs_value[X["feature"]] = [v_x]
            xs_value[Y["feature"]] = [v_y]
            for feature in input_features:
                input_tmp += xs_value[feature]
            inputs.append(input_tmp)
    Z = {
        "dynamics": {
            "name": z["feature"], 
            MEAN: [], SIGMA: [], 
            "range": z["range"]}, 
        "reward": {
            "name": reward_function["name"], 
            MEAN: [], SIGMA: [], 
            "range": reward_function["range"]}
        }
    for input in inputs:
        pred = model.Predict(input, with_var=True)
        output_idx = z["output_dim"]
        Z["dynamics"][MEAN].append(pred.Y[output_idx])
        Z["dynamics"][SIGMA].append(np.sqrt(pred.Var[output_idx,output_idx]))
        input_mean = []
        input_var = []
        for feature in reward_function["input_features"]:
            if feature == z["feature"]:
                input_mean += reward_function["format_mean"](pred)
                input_var += reward_function["format_var"](pred)
            else:
                input_mean += xs_value[feature]
                input_var += [0.]
        pred = reward_function["model"].Predict(x=input_mean, x_var=input_var, with_var=True)
        Z["reward"][MEAN].append(pred.Y[0])
        Z["reward"][SIGMA].append(np.sqrt(pred.Var[0,0]))
    for k_Z in Z.keys():
        for stat_type in [MEAN, SIGMA]:
            Z[k_Z][stat_type] = np.array(Z[k_Z][stat_type]).reshape(len(Y["values"]),len(X["values"]))
    
    footnote = "model_path: "+model_path+"<br>"
    del xs_value[X["feature"]], xs_value[Y["feature"]]
    input_features.remove(X["feature"]), input_features.remove(Y["feature"])
    added_text = ""
    for feature in input_features:
        tmp = added_text + feature + ": " + "[" +', '.join(map(str, xs_value[feature])) + "]" +" "*20
        if len(tmp) >= 140:
            footnote = footnote + tmp + "<br>"
            added_text = ""
        else:
            added_text = tmp   
    footnote += added_text

    for k_Z, v_Z in Z.items():
        for stat_type in [MEAN, SIGMA]:
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=v_Z[stat_type], x=X["values"], y=Y["values"],
                                    colorscale='Oranges',
                                    zmin=v_Z["range"][stat_type][0], 
                                    zmax=v_Z["range"][stat_type][1], 
                                    zauto=False,
                                    colorbar=dict(
                                    title=v_Z["name"]+" ("+stat_type+")",
                                    titleside="top",
                                    #   tickmode="array",
                                    #   tickvals=[2, 1.3, 1, 0, -1],
                                    #   ticktext=["-0.01", "-0.05", "-0.1", "-1", "-10"],
                                    ticks="outside",
                                    )
                                    ))
            fig.update_layout(
                title_text="<b>"+v_Z["name"]+" ("+stat_type+")<b>",
                height=800, width=800, 
                xaxis={"title": X["feature"]}, 
                yaxis={"title": Y["feature"]},
                margin=dict(b=250),
            )
            fig.add_annotation(dict(font=dict(color='black',size=15),
                                                x=0,
                                                y=-0.3,
                                                # showarrow=False,
                                                text=footnote,
                                                # textangle=0,
                                                xanchor='left',
                                                xref="paper",
                                                yref="paper"
                                                ))
            fig.show()
            check_or_create_dir(save_dir)
            plotly.offline.plot(fig, filename = save_dir + file_name_pref + k_Z + "_" + stat_type + "_" + X["feature"].replace("_","") + "_" + Y["feature"].replace("_","") + "_" + z["feature"] + ".html", auto_open=False)