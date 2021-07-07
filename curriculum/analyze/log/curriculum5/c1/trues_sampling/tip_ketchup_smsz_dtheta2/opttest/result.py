# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .learn import *


BASE_DIR = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/"

def Run(ct, *args):
    name = "Er"
    trial_list = ["t1","t2","t3"]
    save_img_dir = PICTURE_DIR + "opttest/onpolicy/{}_n{}/".format(name.replace("/","_"), len(trial_list))
    
    results = []
    for trial in trial_list:
        logdir = BASE_DIR + "opttest/logs/onpolicy/{}/{}/".format(name, trial)
        dm = Domain.load(logdir+"dm.pickle")
        results.append(dm.log["true_r_at_est_opt_dthtea2"])
    results_mean = np.mean(np.array(results).T, axis=1)
    results_sd = np.std(np.array(results).T, axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = np.linspace(0,len(results_mean)), y = results_mean,
        mode='markers',
        error_y=dict(
            array = results_sd
        ) 
    ))
    fig['layout']['xaxis']['title'] = "episode"
    fig['layout']['yaxis']['title'] = "return"
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "result.html", auto_open=False)