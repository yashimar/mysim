# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .check import *


def Run(ct, *args):
    name_pref = "t0.1"
    name_list = [
        "t0.1/t1",
        "t0.1/t2",
        "t0.1/t3",
        "t0.1/t4",
        "t0.1/t5",
        "t0.1/t6",
        "t0.1/t7",
        "t0.1/t8",
        "t0.1/t9",
        "t0.1/t10",
    ]
    save_img_dir = PICTURE_DIR + "opttest/{}/".format(name_pref.replace("/","_"))

    datotal_meta, reward_meta = [], []
    for name in name_list:
        logdir = BASE_DIR + "opttest/logs/{}/".format(name)
        print(logdir)
        dm = Domain.load(logdir+"dm.pickle")
        gmm1 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50], Gerr = 1.0)
        gmm1.train()
        gmm2 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50], Gerr = 2.0)
        gmm2.train()
        
        if os.path.exists(logdir+"datotal.pickle"):
            with open(logdir+"datotal.pickle", mode="rb") as f:
                datotal = pickle.load(f)
            with open(logdir+"reward.pickle", mode="rb") as f:
                reward = pickle.load(f)
        else:
            datotal, reward = setup(dm, gmm1, gmm2, logdir)
        datotal_meta.append(datotal)
        reward_meta.append(reward)

    
    #最適化された評価関数曲線
    print("最適化された評価関数曲線")
    r_types = (Er, Er_1LCB, Er_2LCB)
    fig = go.Figure()
    for r_type in r_types:
        opt_dtheta2_list_meta = []
        for reward in reward_meta:
            opt_dtheta2_list = np.argmax(reward[r_type], axis = 0)
            opt_dtheta2_list_meta.append(opt_dtheta2_list)
        y_list_meta = [[smsz_r[opt_idx] for smsz_r, opt_idx in zip(dm.datotal[RFUNC].T, opt_dtheta2_list)] for opt_dtheta2_list in opt_dtheta2_list_meta]
        fig.add_trace(
            go.Scatter(
                x=dm.smsz, y=np.mean(y_list_meta, axis=0),
                mode='markers', 
                name=r_type+" [True Value]",
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=np.std(y_list_meta, axis=0),
                    thickness=1.5,
                    width=3,
                )
            )
        )
        y_list_meta = [[smsz_r[opt_idx] for smsz_r, opt_idx in zip(reward[r_type].T, opt_dtheta2_list)] for opt_dtheta2_list in opt_dtheta2_list_meta]
        fig.add_trace(
            go.Scatter(
                x=dm.smsz, y=np.mean(y_list_meta, axis=0),
                mode='lines', 
                name=r_type+" [Est Opt]",
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=np.std(y_list_meta, axis=0),
                    thickness=1.5,
                    width=3,
                )
            )
        )
    fig['layout']['yaxis']['range'] = (-8,0.1)
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "return"
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "opt_return.html", auto_open=False)