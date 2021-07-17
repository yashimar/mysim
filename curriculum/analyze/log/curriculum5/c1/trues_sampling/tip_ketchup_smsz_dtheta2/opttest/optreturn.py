# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .setup import *


def Run(ct, *args):
    name_pref = "t0.1/1000"
    name_list = [
        "t0.1/1000/t1",
        "t0.1/1000/t2",
        "t0.1/1000/t3",
        "t0.1/1000/t4",
        "t0.1/1000/t5",
        "t0.1/1000/t6",
        "t0.1/1000/t7",
        "t0.1/1000/t8",
        "t0.1/1000/t9",
        "t0.1/1000/t10",
        "t0.1/1000/t11",
        "t0.1/1000/t12",
        "t0.1/1000/t13",
        "t0.1/1000/t14",
        "t0.1/1000/t15",
        "t0.1/1000/t16",
        "t0.1/1000/t17",
        "t0.1/1000/t18",
        "t0.1/1000/t19",
        "t0.1/1000/t20",
        "t0.1/1000/t21",
        "t0.1/1000/t22",
        "t0.1/1000/t23",
        "t0.1/1000/t24",
        "t0.1/1000/t25",
        "t0.1/1000/t26",
        "t0.1/1000/t27",
        "t0.1/1000/t28",
        "t0.1/1000/t29",
        "t0.1/1000/t30",
        "t0.1/1000/t31",
        "t0.1/1000/t32",
        "t0.1/1000/t33",
        "t0.1/1000/t34",
        "t0.1/1000/t35",
        "t0.1/1000/t36",
        "t0.1/1000/t37",
        "t0.1/1000/t38",
        "t0.1/1000/t39",
        "t0.1/1000/t40",
    ]
    save_img_dir = PICTURE_DIR + "opttest/{}/".format(name_pref)
    recreate = False

    datotal_meta, reward_meta = [], []
    for name in name_list:
        print(name)
        logdir = BASE_DIR + "opttest/logs/{}/".format(name)
        dm = Domain.load(logdir+"dm.pickle")
        observations = np.array([
            dm.log["est_opt_dtheta2"],
            dm.log["smsz"]
        ]).T
        GMM3Sig001 = GMM3(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
        GMM3Sig001.train()
        GMM3Sig002 = GMM3(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50], Gerr = 1.0)
        GMM3Sig002.train()
        GMM3Sig003 = GMM3(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/33.3, (max(dm.smsz)-min(dm.smsz))/33.3], Gerr = 1.0)
        GMM3Sig003.train()
        GMM3Sig005 = GMM3(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
        GMM3Sig005.train()
        CGMMSig001Pt09 = CGMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
        CGMMSig001Pt09.train(p_thr=0.9)
        CGMMSig003Pt09 = CGMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/33.3, (max(dm.smsz)-min(dm.smsz))/33.3], Gerr = 1.0)
        CGMMSig003Pt09.train(p_thr=0.9)
        CGMMSig005Pt09 = CGMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
        CGMMSig005Pt09.train(p_thr=0.9)
        gmm_name_list = [
                # (GMM2Sig001, "GMM2Sig001"),
                # (GMM2Sig002, "GMM2Sig002"),
                # (GMM2Sig003, "GMM2Sig003"),
                # (GMM2Sig005, "GMM2Sig005"),
                (GMM3Sig001, "GMM3Sig001"),
                # (GMM3Sig002, "GMM3Sig002"),
                (GMM3Sig003, "GMM3Sig003"),
                (GMM3Sig005, "GMM3Sig005"),
                
                # (CGMMSig001Pt09, "CGMMSig001Pt09"),
                # (CGMMSig003Pt09, "CGMMSig003Pt09"),
                # (CGMMSig005Pt09, "CGMMSig005Pt09"),
        ]
        datotal = setup_datotal(dm, logdir)
        gmmpred = setup_gmmpred(dm, gmm_name_list, logdir)
        gmm_names = [gmm_name for _, gmm_name in gmm_name_list]
        gain_pairs = [
            # (1.0,0.2), 
            (1.0,0.5), 
            (1.0,1.0), 
        ]
        reward = setup_reward(dm, logdir, gmm_names, gain_pairs)
        
        datotal_meta.append(datotal)
        reward_meta.append(reward)

    
    #最適化された評価関数曲線
    print("最適化された評価関数曲線")
    r_types = [Er, "Er_LCB2"]
    gains = ["gnnsd{}_ggmm{}".format(gnnsd, ggmm) for gnnsd, ggmm in gain_pairs]
    gmm_types = deepcopy(gmm_names)
    r_types += ["{}_{}~{}".format(gmm_type, gain, r_type) for r_type in r_types for gmm_type in gmm_types for gain in gains]
    fig = go.Figure()
    for name in r_types:
        opt_dtheta2_list_meta = []
        for reward in reward_meta:
            opt_dtheta2_list = np.argmax(reward[name], axis = 0)
            opt_dtheta2_list_meta.append(opt_dtheta2_list)
        y_list_meta = [[smsz_r[opt_idx] for smsz_r, opt_idx in zip(dm.datotal[RFUNC].T, opt_dtheta2_list)] for opt_dtheta2_list in opt_dtheta2_list_meta]
        text = ["<br />".join(["t{}: {}".format(j+1,y_list[i]) for j,y_list in enumerate(y_list_meta)]) for i in range(len(dm.smsz))]
        fig.add_trace(
            go.Scatter(
                x=dm.smsz, y=np.mean(y_list_meta, axis=0),
                mode='markers', 
                name="{}".format(name),
                text = text,
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=np.std(y_list_meta, axis=0),
                    thickness=1.5,
                    width=3,
                )
            )
        )
            # y_list_meta = [[smsz_r[opt_idx] for smsz_r, opt_idx in zip(reward[r_type][name].T, opt_dtheta2_list)] for opt_dtheta2_list in opt_dtheta2_list_meta]
            # fig.add_trace(
            #     go.Scatter(
            #         x=dm.smsz, y=np.mean(y_list_meta, axis=0),
            #         mode='lines', 
            #         name=r_type+" [Est Opt]",
            #         error_y=dict(
            #             type="data",
            #             symmetric=True,
            #             array=np.std(y_list_meta, axis=0),
            #             thickness=1.5,
            #             width=3,
            #         )
            #     )
            # )
    fig['layout']['yaxis']['range'] = (-8,0.1)
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "return"
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "opt_return2.html", auto_open=False)
