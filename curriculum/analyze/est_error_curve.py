from core_tool import *
from util import *
from ..tasks_domain.flow_ctrl import task_domain as td

L_MEAN = "latent_mean"
L_SIGMA = "latent_sigma"


def Help():
    pass


def Run(ct, *args):
    log_name_list = [
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c0_init_50",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c1_small_5",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c2_small_nobounce_tip_dtheta2_3",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c3_small_5",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c4_small_ketchup_5",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c5_middle_nobounce_5",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c6_middle_ketchup_5",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c7_large_nobounce_5",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5_5",
        "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1/c8_large_nobounce_tip_5_5_5_5",
        # "curriculum/pouring3/full_scratch/curriculum_test/t1/first150",
    ]
    save_dir = PICTURE_DIR + "curriculum/flow_ctrl/c_adaptive/curriculum_test/t1".replace("/","_") + "/"
    file_name_pref = ""
    dynamics_outdim_pair = {"Ftip": 4,}
    vis_state_dynamics_outdim_lim_pair = [
        ("da_total_tip", "Ftip", 0, (-0.1,0.9)),
        ("lp_flow_x_tip", "Ftip", 1, (-0.3,1.5)),
        ("lp_flow_z_tip", "Ftip", 2, (-0.1,0.1)),
        ("flow_var_tip", "Ftip", 3, (0.1,1.0)),
    ]
    
    domain = td.Domain()
    mm = ModelManager(domain, ROOT_PATH+log_name_list[-1])
    
    pred_true_history = defaultdict(lambda: defaultdict(lambda: {MEAN: [], SIGMA: [], TRUE: [], L_MEAN: [], L_SIGMA: []}))
    for log_name in log_name_list:
        with open(ROOT_PATH+log_name+"/pred_true_log.yaml") as f:
            pred_true_log = yaml.safe_load(f)
        
        for i in range(len(pred_true_log)):
            for dynamics, outdim in dynamics_outdim_pair.items():
                if dynamics in pred_true_log[i].keys():
                    xs = pred_true_log[i][dynamics]
                    latent_model = mm.Models[dynamics][2]
                    latent_p = latent_model.Predict(xs["input"], with_var=True)
                    for j in range(outdim):
                        pred_true_history[dynamics]["out{}".format(j)][MEAN].append(xs["prediction"]["X"][j][0])
                        pred_true_history[dynamics]["out{}".format(j)][SIGMA].append(math.sqrt(xs["prediction"]["Cov"][j][j]))
                        pred_true_history[dynamics]["out{}".format(j)][L_MEAN].append(latent_p.Y[j].item())
                        pred_true_history[dynamics]["out{}".format(j)][L_SIGMA].append(latent_p.Var[j,j].item())
                        pred_true_history[dynamics]["out{}".format(j)][TRUE].append(xs["true_output"][j])
                else:
                    for j in range(outdim):
                        pred_true_history[dynamics]["out{}".format(j)][MEAN].append(np.nan)
                        pred_true_history[dynamics]["out{}".format(j)][SIGMA].append(np.nan)
                        pred_true_history[dynamics]["out{}".format(j)][L_MEAN].append(np.nan)
                        pred_true_history[dynamics]["out{}".format(j)][L_SIGMA].append(np.nan)
                        pred_true_history[dynamics]["out{}".format(j)][TRUE].append(np.nan)
                        
    features = dict()
    for state, dynamics, outdim, _ in vis_state_dynamics_outdim_lim_pair:
        for stat_type in [MEAN, SIGMA, L_MEAN, L_SIGMA]:
            features["{}_pred {}".format(stat_type, state)] = pred_true_history[dynamics]["out{}".format(outdim)][stat_type]
        features["true {}".format(state)] = pred_true_history[dynamics]["out{}".format(outdim)][TRUE]
        features["episode"] = np.arange(0,len(pred_true_history["Ftip"]["out0"][TRUE]))
    df = pd.DataFrame(features)
    df.dropna(inplace=True)

    for state, dynamics, _, lim in vis_state_dynamics_outdim_lim_pair:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = df["episode"], y=df["mean_pred {}".format(state)],
            mode='markers',
            name='pred mean+/-sigma',
            error_y=dict(
                type="data",
                symmetric=True,
                array=df["sigma_pred {}".format(state)].values.tolist(),
                color='orange',
                thickness=1.5,
                width=3,
            ),
            marker=dict(color='orange', size=8)
        ))
        fig.add_trace(go.Scatter(
            x = df["episode"], y=df["latent_mean_pred {}".format(state)],
            mode='markers',
            name='latent_pred mean+/-sigma',
            error_y=dict(
                type="data",
                symmetric=True,
                array=df["latent_sigma_pred {}".format(state)].values.tolist(),
                color='purple',
                thickness=1.5,
                width=3,
            ),
            marker=dict(color='purple', size=8)
        ))
        fig.add_trace(go.Scatter(
            x = df["episode"], y=df["true {}".format(state)],
            mode='markers',
            name='true',
            marker=dict(color='blue', size=8)
        ))
        fig.update_layout(
            title="{} {}".format(dynamics, state),
            xaxis_title="episode",
            yaxis_title="{}".format(state),
            yaxis_range=lim,
        )
        fig.show()
        check_or_create_dir(save_dir)
        plotly.offline.plot(fig, filename = save_dir + file_name_pref + dynamics + "_" + state.replace("_","") + ".html", auto_open=False)