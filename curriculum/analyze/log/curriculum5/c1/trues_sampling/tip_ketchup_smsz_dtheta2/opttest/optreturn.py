# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .setup import *


def Run(ct, *args):
    name_pref = "t0.1"
    name_list = [
        "t0.1/500/t1",
        "t0.1/500/t2",
        "t0.1/500/t3",
        "t0.1/500/t4",
        "t0.1/500/t5",
        "t0.1/500/t6",
        "t0.1/500/t7",
        "t0.1/500/t8",
        "t0.1/500/t9",
        "t0.1/500/t10",
        "t0.1/500/t11",
        "t0.1/500/t12",
        "t0.1/500/t13",
        "t0.1/500/t14",
        "t0.1/500/t15",
        "t0.1/500/t16",
        "t0.1/500/t17",
        "t0.1/500/t18",
        "t0.1/500/t19",
        "t0.1/500/t20",
    ]
    save_img_dir = PICTURE_DIR + "opttest/{}/".format(name_pref.replace("/","_"))
    recreate = False

    datotal_meta, reward_meta = [], []
    for name in name_list:
        logdir = BASE_DIR + "opttest/logs/{}/".format(name)
        dm = Domain.load(logdir+"dm.pickle")
        observations = np.array([
            dm.log["est_opt_dtheta2"],
            dm.log["smsz"]
        ]).T
        # G1ObsrSig001 = ObservationReward(observations, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100])
        # G1ObsrSig001.setup()
        # G1ObsrSig002 = ObservationReward(observations, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50])
        # G1ObsrSig002.setup()
        # G1ObsrSig005 = ObservationReward(observations, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20])
        # G1ObsrSig005.setup()
        obsr_name_list = [
            # (G1ObsrSig001, "G1ObsrSig001"),
            # (G1ObsrSig002, "G1ObsrSig002"),
            # (G1ObsrSig005, "G1ObsrSig005"),
        ]
        S005unobsSig001 = UnobservedSD(observations, penalty=0.05, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100])
        S005unobsSig001.setup()
        # S01unobsSig001 = UnobservedSD(observations, penalty=0.05, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100])
        # S01unobsSig001.setup()
        # S01unobsSig002 = UnobservedSD(observations, penalty=0.1, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50])
        # S01unobsSig002.setup()
        # S03unobsSig001 = UnobservedSD(observations, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100])
        # S03unobsSig001.setup()
        # S03unobsSig002 = UnobservedSD(observations, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50])
        # S03unobsSig002.setup()
        unobs_name_list = [
            (S005unobsSig001, "S005unobsSig001"),
            # (S01unobsSig001, "S01unobsSig001"),
            # (S01unobsSig002, "S01unobsSig002"),
            # (S03unobsSig001, "S03unobsSig001"),
            # (S03unobsSig002, "S03unobsSig002"),
        ]
        Gerr1Sig001 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
        Gerr1Sig001.train()
        Gerr1Sig002 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50], Gerr = 1.0)
        Gerr1Sig002.train()
        # Gerr1Sig005 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
        # Gerr1Sig005.train()
        GMM2Sig001 = GMM2(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
        GMM2Sig001.train()
        GMM2Sig003 = GMM2(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/33.3, (max(dm.smsz)-min(dm.smsz))/33.3], Gerr = 1.0)
        GMM2Sig003.train()
        GMM2Sig005 = GMM2(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
        GMM2Sig005.train()
        gmm_name_list = [
            # (Gerr1Sig001, "Gerr1_Sig001"),
            # (Gerr1Sig002, "Gerr1_Sig002"),
            # (Gerr1Sig005, "Gerr1_Sig005"),
            (GMM2Sig001, "GMM2Sig001"),
            (GMM2Sig003, "GMM2Sig003"),
            (GMM2Sig005, "GMM2Sig005"),
        ]
        datotal = setup_datotal(dm, logdir)
        reward = setup_reward(dm, logdir)
        datotal_meta.append(datotal)
        reward_meta.append(reward)

    
    #最適化された評価関数曲線
    print("最適化された評価関数曲線")
    r_types = {"normal": [Er, "Er_LCB2"]}
    # r_types = r_types_base + ["{}~{}".format(name, r_type) for r_type in r_types_base for name in unobssds.keys()]
    # r_types = r_types + ["{}~Er_add_LCB2".format(name) for name in gmmpred.keys()]
    # r_types = r_types + ["{}-{}~{}".format(n_unobs, n_gmm, r_type) for r_type in r_types_base for n_unobs in unobssds.keys() for n_gmm in gmmpred.keys()]
    r_types["gmm1"] = ["{}~{}".format(n_gmm, name) for name in r_types["normal"] for _,n_gmm in gmm_name_list]
    r_types["gmm2"] = ["{}~{}".format(n_gmm, name) for name in r_types["normal"] for _,n_gmm in gmm_name_list]
    r_types["gmm3"] = ["{}~{}".format(n_gmm, name) for name in r_types["normal"] for _,n_gmm in gmm_name_list]
    fig = go.Figure()
    for r_type, names in r_types.items():
        for name in names:
            opt_dtheta2_list_meta = []
            for reward in reward_meta:
                opt_dtheta2_list = np.argmax(reward[r_type][name], axis = 0)
                opt_dtheta2_list_meta.append(opt_dtheta2_list)
            y_list_meta = [[smsz_r[opt_idx] for smsz_r, opt_idx in zip(dm.datotal[RFUNC].T, opt_dtheta2_list)] for opt_dtheta2_list in opt_dtheta2_list_meta]
            text = ["<br />".join(["t{}: {}".format(j+1,y_list[i]) for j,y_list in enumerate(y_list_meta)]) for i in range(len(dm.smsz))]
            fig.add_trace(
                go.Scatter(
                    x=dm.smsz, y=np.mean(y_list_meta, axis=0),
                    mode='markers', 
                    name="{} {}".format(r_type, name),
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
    plotly.offline.plot(fig, filename = save_img_dir + "opt_return.html", auto_open=False)
