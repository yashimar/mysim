# coding: UTF-8
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from ..tasks_domain.before_normalize.util import ModelManager as PrevModelManager
from ..tasks_domain.util import ModelManager3 as ModelManager
# from ...fixed_script.ml_dnn import TNNRegression2
# from ...fixed_script.dpl4 import TModelManager3
import os
from collections import defaultdict
import yaml
from yaml.representer import Representer
yaml.add_representer(defaultdict, Representer.represent_dict)
import pickle
import joblib
import glob
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2



ROOT_PATH = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
PICTURE_DIR = "/home/yashima/Pictures/"
MEAN = "mean"
SIGMA = "sigma"
L_MEAN = "latent_mean"
L_SIGMA = "latent_sigma"
FORCE_TO_NONE = "froce_to_None"
DEAL_AS_ZERO = "deal_as_zero"
STAT_TYPES = [MEAN, SIGMA]
ERROR = "err"
MODEL_TYPES = [MEAN, ERROR]
PLUS = "plus"
MINUS = "minus"
TRUE = "true"
NOBOUNCE = [0.1, 0.2, 0.0, 0.1]
KETCHUP = [0.1, 0.01, 0.25, 0.2]

def get_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate = False, root_path = ROOT_PATH):
    sh_path = root_path + save_sh_dir + "/state_history.yaml"
    
    if (recreate == False) & (os.path.exists(sh_path)):
        with open(sh_path) as f:
            state_histories = yaml.safe_load(f)["Seq"]
            
    else:
        state_histories = defaultdict(lambda: defaultdict(lambda: {MEAN: [], SIGMA: []}))
        for log_name in log_name_list:
            db_path = root_path + log_name + "/database.yaml"
            with open(db_path) as f:
                db = yaml.safe_load(f)["Entry"]
            
            for seq in db:
                seq = seq["Seq"]
                for node, state_dim_set in node_states_dim_pair:
                    for state, dim in state_dim_set:
                        is_found_node = False
                        for node_xs in seq:
                            if node == node_xs["Name"]:
                                is_found_node = True
                                XS = node_xs["XS"][state]
                                if dim == 1:
                                    state_histories[node][state][MEAN].append(XS["X"][0][0])
                                    state_histories[node][state][SIGMA].append(np.sqrt(XS["Cov"][0][0]).item() if XS["Cov"] is not None else None)
                                else:
                                    for i, x in enumerate(XS["X"]):
                                        state_histories[node][state+"_"+str(i)][MEAN].append(x[0])
                                        state_histories[node][state+"_"+str(i)][SIGMA].append(np.sqrt(np.diag(XS["Cov"])[i]).item() if XS["Cov"] is not None else None)
                                break
                        if is_found_node == False:
                            if dim == 1:
                                state_histories[node][state][MEAN].append(None)
                                state_histories[node][state][SIGMA].append(None)
                            else:
                                for i, x in enumerate(XS["X"]):
                                    state_histories[node][state+"_"+str(i)][MEAN].append(None)
                                    state_histories[node][state+"_"+str(i)][SIGMA].append(None)
            
        with open(sh_path, "w") as f:
            yaml.dump({"Seq": state_histories}, f, default_flow_style=False)
                            
    return state_histories


def get_bestpolicy_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate = False, root_path = ROOT_PATH):
    esh_path = root_path + save_sh_dir + "/bestpolicy_est_state_history.yaml"
    
    if (recreate == False) & (os.path.exists(esh_path)):
        with open(esh_path) as f:
            est_state_histories = yaml.safe_load(f)["Seq"]
    
    else:
        est_state_histories = defaultdict(lambda: defaultdict(lambda: {MEAN: [], SIGMA: []}))
        for log_name in log_name_list:
            tree_path = root_path + log_name + "/best_est_trees" + "/"
            for i in range(len(os.listdir(tree_path))):
                with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
                    tree = joblib.load(f).Tree
                    for node, state_dim_set in node_states_dim_pair:
                        for state, dim in state_dim_set:
                            XS = tree[TPair(node,0)].XS[state]
                            if dim == 1:
                                est_state_histories[node][state][MEAN].append(XS.X.item())
                                est_state_histories[node][state][SIGMA].append(np.sqrt(XS.Cov).item() if XS.Cov is not None else None)
                            else:
                                for i, x in enumerate(XS.X):
                                    est_state_histories[node][state+"_"+str(i)][MEAN].append(x.item())
                                    est_state_histories[node][state+"_"+str(i)][SIGMA].append(np.sqrt(np.diag(XS.Cov)[i]).item() if XS.Cov is not None else None)
                        
        with open(esh_path, "w") as f:
            yaml.dump({"Seq": est_state_histories}, f, default_flow_style=False)
                
    return est_state_histories


def get_true_and_bestpolicy_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate = False, root_path = ROOT_PATH):
    sh = get_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate)
    esh = get_bestpolicy_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate)
    
    return sh, esh


def operate_list(d_lists, operator=PLUS, deal_with_None=FORCE_TO_NONE):
    s = 0
    none_index_list = False
    for d_list in d_lists:
        array = np.array(d_list).astype(np.float32)
        none_index_list += np.isnan(array)
        if operator == PLUS:
            s += np.nan_to_num(array, 0.)
        elif operator == MINUS:
            s -= np.nan_to_num(array, 0.)
    if deal_with_None == FORCE_TO_NONE:
        np.place(s, none_index_list, np.nan)
    elif deal_with_None == DEAL_AS_ZERO:
        pass
    else:
        raise(Exception("deal_with_None is not valid."))
    
    return s.tolist()


def check_or_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        

def base_trace(x, y, color, text):
    trace = go.Scatter(
        x=x, y=y, 
        mode='markers', 
        # marker_color="blue",
        opacity = 0.5,
        hoverinfo='text',
        text=text,
        showlegend=False,
        marker = dict(
            size = 10,
            color = color,
            colorscale="Viridis",
            # cmin = 0,
            # cmax = 0.55,
        ),
    )
    return trace


def plot_and_save_df_scatter(vis_df_title_pair, xy_limit_pairs, save_img_dir, concat_title, go_layout, color=None, updatemenu=None):
    n_graph = len(xy_limit_pairs)
    color = color if color != None else lambda df: df.index
    fig = make_subplots(
            rows=n_graph, cols=1,
            # subplot_titles=["[{}] {} / {}".format(title, x, y) for x, y, _, _ in xy_limit_pairs for _, title in vis_df_title_pair],
            # horizontal_spacing = 0.1,
            # vertical_spacing = 0.1,
        )
    fig.update_layout(**go_layout)
    for xy_i, (x, y, xlim, ylim) in enumerate(xy_limit_pairs):
        r_idx = xy_i+1
        for df, _ in vis_df_title_pair:
            text = ["".join(["{}: {}<br />".format(c, df[c][i]) for c in df.columns if c!="comment"])+("<b>comment</b>: {}".format(df["comment"][i]) if df["comment"][i] != "" else "") for i in df.index]
            fig.add_trace(base_trace(df[x],df[y],color(df),text), r_idx, 1)
            fig['layout']['xaxis{}'.format(r_idx)]['title'] = x
            fig['layout']['yaxis{}'.format(r_idx)]['title'] = y
            fig['layout']['xaxis{}'.format(r_idx)]['range'] = xlim
            fig['layout']['yaxis{}'.format(r_idx)]['range'] = ylim
    if updatemenu != None:
        updatemenu(fig)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir+"{}.html".format(concat_title), auto_open=False)
    
    
def pred_test(model):
    for x,t in zip(model.DataX, model.DataY):
        p = model.Predict(x, with_var=True)
        print("---")
        Print("input:", x)
        Print("est:", p.Y[0].item())
        Print("sigma:", np.sqrt(p.Var[0,0]))
        Print("true:", t[0])
        Print("diff:", abs(p.Y[0].item()-t[0]))
        

# def learn_test(model):
    # y = model.Forward(model.DataX[0:1], True)
    # t = Variable(model.DataY[0:1])
    # loss = F.mean_squared_error(y, t)
    # loss.backward(retain_grad=True)
    # print(loss.requires_grad)
    

def plot_dynamics_heatmap(td, model_path, save_dir, file_name_pref, model_name, xs_value, input_features, X, Y, z, reward_function, scatter_obj_list=None, updatemenu=None, model=None, is_prev_model=False):
    if model == None:
        domain = td.Domain()
        if is_prev_model:
            mm = PrevModelManager(domain, ROOT_PATH+model_path)
        else:
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
    
    def make_hovertext(Z, stat_type):
        hovertext = []
        for i, y in enumerate(Y["values"]):
            hovertext.append([])
            for j, x, in enumerate(X["values"]):
                hovertext[-1].append('{}: {}<br />{}: {}<br />{}: {}'.format(Z["name"]+" ("+stat_type+")",Z[stat_type][i][j],X["feature"],x,Y["feature"],y))
        return hovertext
    
    subplot_titles = ["<b>"+v_Z["name"]+" ("+stat_type+")<b>" for v_Z in Z.values() for stat_type in STAT_TYPES]
    
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=subplot_titles,
        horizontal_spacing = 0.1,
        vertical_spacing = 0.1,
    )
    fig.update_layout(
                height=950, width=1800, 
                margin=dict(t=20,b=150),
                hoverdistance = 5,
            )
    x1,x2,y1,y2 = 0.46,1.0075,0.78,0.22
    colorbar_loc = [
        [[x1,y1],[x2,y1]],
        [[x1,y2],[x2,y2]]
    ]
    r = 0
    i = 0
    for k_Z, v_Z in Z.items():
        r += 1
        c = 0
        for stat_type in STAT_TYPES:
            c += 1
            i += 1
            fig.add_trace(go.Heatmap(z=v_Z[stat_type], x=X["values"], y=Y["values"],
                                    colorscale='Oranges',                 
                                    zmin=v_Z["range"][stat_type][0], zmax=v_Z["range"][stat_type][1], zauto=False,
                                    colorbar=dict(
                                        title=v_Z["name"]+" ("+stat_type+")",
                                        titleside="top", ticks="outside",
                                        #   tickmode="array",
                                        #   tickvals=[2, 1.3, 1, 0, -1],
                                        #   ticktext=["-0.01", "-0.05", "-0.1", "-1", "-10"],
                                        x = colorbar_loc[r-1][c-1][0], y = colorbar_loc[r-1][c-1][1],
                                        thickness=23, len = 0.45,
                                    ),
                                    hoverinfo='text',
                                    text=make_hovertext(v_Z, stat_type),
                        ), r, c)
            if scatter_obj_list != None:
                for scatter_obj in scatter_obj_list:
                    fig.add_trace(scatter_obj, r, c)
            fig['layout']['xaxis'+str(i)]['title'] = X["feature"]
            fig['layout']['yaxis'+str(i)]['title'] = Y["feature"]
    fig.add_annotation(dict(font=dict(color='black',size=15),
                                                x=0,
                                                y=-0.15,
                                                # showarrow=False,
                                                text=footnote,
                                                # textangle=0,
                                                xanchor='left',
                                                xref="paper",
                                                yref="paper"
                                                ))
    if updatemenu != None:
        updatemenu(fig)
    check_or_create_dir(save_dir)
    plotly.offline.plot(fig, filename = save_dir + file_name_pref + X["feature"].replace("_","") + "_" + Y["feature"].replace("_","") + "_" + z["feature"].replace("_","") + ".html", auto_open=False)
    

def remake_model(td, model_name, model_path, save_path, suff):
    mm = ModelManager(td.Domain(), ROOT_PATH+model_path)
    model = mm.Models[model_name][2]
    model.Options["base_dir"] = ROOT_PATH+save_path+"/"+suff+"/models/"
    DataX, DataY = model.DataX, model.DataY
    for key in ["nn_params", "nn_params_err", "nn_data_x", "nn_data_y"]:
        model.Params[key] = None
    model.Params["num_train"] = 0
    model.Init()
    
    return model, DataX, DataY

def transition_plot(td, log_name_list, dynamics_iodim_pair, vis_state_dynamics_outdim_lim_pair, go_layout, suff_annotation, save_dir, file_name_pref, vis_condition_title_pair, updatemenu):
    domain = td.Domain()
    mm = ModelManager(domain, ROOT_PATH+log_name_list[-1])
    
    pred_true_history = defaultdict(lambda: defaultdict(lambda: {MEAN: [], SIGMA: [], TRUE: [], L_MEAN: [], L_SIGMA: []}))
    for log_name in log_name_list:
        with open(ROOT_PATH+log_name+"/pred_true_log.yaml") as f:
            pred_true_log = yaml.safe_load(f)
        
        for i in range(len(pred_true_log)):
            for dynamics, (indim, outdim) in dynamics_iodim_pair.items():
                if dynamics in pred_true_log[i].keys():
                    xs = pred_true_log[i][dynamics]
                    latent_model = mm.Models[dynamics][2]
                    latent_p = latent_model.Predict(xs["input"], with_var=True)
                    for j in range(outdim):
                        pred_true_history[dynamics]["out{}".format(j)][MEAN].append(xs["prediction"]["X"][j][0])
                        pred_true_history[dynamics]["out{}".format(j)][SIGMA].append(math.sqrt(xs["prediction"]["Cov"][j][j]))
                        pred_true_history[dynamics]["out{}".format(j)][L_MEAN].append(latent_p.Y[j].item())
                        pred_true_history[dynamics]["out{}".format(j)][L_SIGMA].append(np.sqrt(latent_p.Var[j,j]).item())
                        pred_true_history[dynamics]["out{}".format(j)][TRUE].append(xs["true_output"][j])
                    for j in range(indim):
                        pred_true_history[dynamics]["in{}".format(j)][TRUE].append(xs["input"][j])
                else:
                    for j in range(outdim):
                        pred_true_history[dynamics]["out{}".format(j)][MEAN].append(np.nan)
                        pred_true_history[dynamics]["out{}".format(j)][SIGMA].append(np.nan)
                        pred_true_history[dynamics]["out{}".format(j)][L_MEAN].append(np.nan)
                        pred_true_history[dynamics]["out{}".format(j)][L_SIGMA].append(np.nan)
                        pred_true_history[dynamics]["out{}".format(j)][TRUE].append(np.nan)
                    for j in range(indim):
                        pred_true_history[dynamics]["in{}".format(j)][TRUE].append(np.nan)
                        
    features = dict()
    for state, dynamics, outdim, _ in vis_state_dynamics_outdim_lim_pair:
        for stat_type in [MEAN, SIGMA, L_MEAN, L_SIGMA]:
            features["{}_pred {}".format(stat_type, state)] = pred_true_history[dynamics]["out{}".format(outdim)][stat_type]
        features["true {}".format(state)] = pred_true_history[dynamics]["out{}".format(outdim)][TRUE]
        features["episode"] = np.arange(0,len(pred_true_history["Ftip"]["out0"][TRUE]))
    # df = pd.DataFrame(features)
    # df.dropna(inplace=True)

    fig = make_subplots(
        rows=2*len(vis_state_dynamics_outdim_lim_pair), cols=1,
        subplot_titles=sum([["{} {}".format(dynamics, state), "{} {} est_error".format(dynamics, state)] for dynamics, state, _, _ in vis_state_dynamics_outdim_lim_pair],[]),
        # horizontal_spacing = 0.5,
        # specs = [[{},] for _ in range(2*len(vis_state_dynamics_outdim_lim_pair))],
        # vertical_spacing = 0.05,
    )
    go_layout.update({'annotations': [{"xanchor": "center", "opacity": 0.8, 'bordercolor': "rgba(0,0,0,0)", 'bgcolor': "rgba(0,0,0,0)"}]})
    fig.update_layout(**go_layout)
    for r, (state, dynamics, _, lim) in enumerate(vis_state_dynamics_outdim_lim_pair):
        for _, condition, _ in vis_condition_title_pair:
            df = pd.DataFrame(features)[condition]
            anno_text = ["<b>true inputs</b><br />"+"".join(["　in{}: {}<br />".format(j, pred_true_history[dynamics]["in{}".format(j)][TRUE][i]) for j in range(dynamics_iodim_pair[dynamics][0])]) for i in df.index]
            anno_text = [t1+t2 for t1,t2 in zip(anno_text,suff_annotation(df))]
            fig.add_trace(go.Scatter(
                x = df["episode"], y=df["mean_pred {}".format(state)],
                mode='markers',
                name='pred mean+/-sigma',
                text = anno_text,
                legendgroup=str(2*r+1),
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=df["sigma_pred {}".format(state)].values.tolist(),
                    color='orange',
                    thickness=1.5,
                    width=3,
                ),
                marker=dict(color='orange', size=8)
            ), 2*r+1, 1)
            fig.add_trace(go.Scatter(
                x = df["episode"], y=df["latent_mean_pred {}".format(state)],
                mode='markers',
                name='latent_pred mean+/-sigma',
                text = anno_text,
                legendgroup=str(2*r+1),
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=df["latent_sigma_pred {}".format(state)].values.tolist(),
                    color='purple',
                    thickness=1.5,
                    width=3,
                ),
                marker=dict(color='purple', size=8)
            ), 2*r+1, 1)
            fig.add_trace(go.Scatter(
                x = df["episode"], y=df["true {}".format(state)],
                mode='markers',
                name='true',
                text = anno_text,
                legendgroup=str(2*r+1),
                marker=dict(color='blue', size=8)
            ), 2*r+1, 1)
            fig['layout']['xaxis{}'.format(2*r+1)]['title'] = "episode"
            fig['layout']['yaxis{}'.format(2*r+1)]['title'] = "{}".format(state)
            fig['layout']['yaxis{}'.format(2*r+1)]['range'] = lim
            
            fig.add_trace(go.Scatter(
                x = df["episode"], y=df["mean_pred {}".format(state)]-df["true {}".format(state)],
                mode='markers',
                name='pred mean - true',
                text = anno_text,
                legendgroup=str(2*r+2),
                marker=dict(color='orange', size=8)
            ), 2*r+2, 1)
            fig.add_trace(go.Scatter(
                x = df["episode"], y=df["latent_mean_pred {}".format(state)]-df["true {}".format(state)],
                mode='markers',
                name='latent_pred mean - true',
                text = anno_text,
                legendgroup=str(2*r+2),
                marker=dict(color='purple', size=8)
            ), 2*r+2, 1)
            fig.add_shape(type='line',
                x0=0,
                x1=max(df["episode"]),
                y0=0,
                y1=0,
                line=dict(color='blue'),
                row=2*r+2,
                col=1,
            )
            fig['layout']['xaxis{}'.format(2*r+2)]['title'] = "episode"
            fig['layout']['yaxis{}'.format(2*r+2)]['title'] = "est_error {}".format(state)
            fig['layout']['yaxis{}'.format(2*r+2)]['range'] = lim
    updatemenu(fig)
    check_or_create_dir(save_dir)
    plotly.offline.plot(fig, filename = save_dir + file_name_pref + "transition.html", auto_open=False)