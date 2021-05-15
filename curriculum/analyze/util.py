from core_tool import *
import os
from collections import defaultdict
import yaml
from yaml.representer import Representer
yaml.add_representer(defaultdict, Representer.represent_dict)
import numpy as np
import joblib
from matplotlib import pyplot as plt
import cv2


ROOT_PATH = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
PICTURE_DIR = "/home/yashima/Pictures/"
MEAN = "mean"
SIGMA = "sigma"
FORCE_TO_NONE = "froce_to_None"
DEAL_AS_ZERO = "deal_as_zero"

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


def get_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate = False, root_path = ROOT_PATH):
    esh_path = root_path + save_sh_dir + "/est_state_history.yaml"
    
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
                            break
                        
        with open(esh_path, "w") as f:
            yaml.dump({"Seq": est_state_histories}, f, default_flow_style=False)
                
    return est_state_histories


def get_true_and_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate = False, root_path = ROOT_PATH):
    sh = get_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate)
    esh = get_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate)
    
    return sh, esh


def plus_list(d_lists, deal_with_None=FORCE_TO_NONE):
    s = 0
    none_index_list = False
    for d_list in d_lists:
        array = np.array(d_list).astype(np.float32)
        none_index_list += np.isnan(array)
        s += np.nan_to_num(array, 0.)
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
        
        
def plot_and_save_df_scatter(df, xy_limit_pairs, save_img_dir, concat_title):
    plt.close("all")
    check_or_create_dir(save_img_dir)
    concat_imgs = []
    for x, y, xlim, ylim in xy_limit_pairs:
        save_path = save_img_dir+x.replace("_","")+"_"+y.replace("_","")+".png"
        df.plot.scatter(x=x,y=y,xlim=xlim,ylim=ylim).get_figure().savefig(save_path)
        concat_imgs.append(cv2.imread(save_path))
    plt.close("all")
    im_v = cv2.vconcat(concat_imgs)
    cv2.imwrite(save_img_dir+concat_title, im_v)
    
    
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