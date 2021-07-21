# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .setup import *


def Run(ct, *args):
    name_pref = "t0.1/200"
    name_list = [
        "t0.1/200/t1",
        "t0.1/200/t2",
        "t0.1/200/t3",
        "t0.1/200/t4",
        "t0.1/200/t5",
        "t0.1/200/t6",
        "t0.1/200/t7",
        "t0.1/200/t8",
        "t0.1/200/t9",
        "t0.1/200/t10",
        "t0.1/200/t11",
        "t0.1/200/t12",
        "t0.1/200/t13",
        "t0.1/200/t14",
        "t0.1/200/t15",
        "t0.1/200/t16",
        "t0.1/200/t17",
        "t0.1/200/t18",
        "t0.1/200/t19",
        "t0.1/200/t20",
        "t0.1/200/t21",
        "t0.1/200/t22",
        "t0.1/200/t23",
        "t0.1/200/t24",
        "t0.1/200/t25",
        "t0.1/200/t26",
        "t0.1/200/t27",
        "t0.1/200/t28",
        "t0.1/200/t29",
        "t0.1/200/t30",
        "t0.1/200/t31",
        "t0.1/200/t32",
        "t0.1/200/t33",
        "t0.1/200/t34",
        "t0.1/200/t35",
        "t0.1/200/t36",
        "t0.1/200/t37",
        "t0.1/200/t38",
        "t0.1/200/t39",
        "t0.1/200/t40",
        "t0.1/200/t41",
        "t0.1/200/t42",
        "t0.1/200/t43",
        "t0.1/200/t44",
        "t0.1/200/t45",
        "t0.1/200/t46",
        "t0.1/200/t47",
        "t0.1/200/t48",
        "t0.1/200/t49",
        "t0.1/200/t50",
        "t0.1/200/t51",
        "t0.1/200/t52",
        "t0.1/200/t53",
        "t0.1/200/t54",
        "t0.1/200/t55",
        "t0.1/200/t56",
        "t0.1/200/t57",
        "t0.1/200/t58",
        "t0.1/200/t59",
        "t0.1/200/t60",
        "t0.1/200/t61",
        "t0.1/200/t62",
        "t0.1/200/t63",
        "t0.1/200/t64",
        "t0.1/200/t65",
        "t0.1/200/t66",
        "t0.1/200/t67",
        "t0.1/200/t68",
        "t0.1/200/t69",
        "t0.1/200/t70",
        "t0.1/200/t71",
        "t0.1/200/t72",
        "t0.1/200/t73",
        "t0.1/200/t74",
        "t0.1/200/t75",
        "t0.1/200/t76",
        "t0.1/200/t77",
        "t0.1/200/t78",
        "t0.1/200/t79",
        "t0.1/200/t80",
        "t0.1/200/t81",
        "t0.1/200/t82",
        "t0.1/200/t83",
        "t0.1/200/t84",
        "t0.1/200/t85",
        "t0.1/200/t86",
        "t0.1/200/t87",
        "t0.1/200/t88",
        "t0.1/200/t89",
        "t0.1/200/t90",
        "t0.1/200/t91",
        "t0.1/200/t92",
        "t0.1/200/t93",
        "t0.1/200/t94",
        "t0.1/200/t95",
        "t0.1/200/t96",
        "t0.1/200/t97",
        "t0.1/200/t98",
        "t0.1/200/t99",
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
        CGMMSig001Pt09 = CGMM(dm.nnmodel, p_thr=0.9, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
        CGMMSig001Pt09.train()
        CGMMSig003Pt09 = CGMM(dm.nnmodel, p_thr=0.9, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/33.3, (max(dm.smsz)-min(dm.smsz))/33.3], Gerr = 1.0)
        CGMMSig003Pt09.train()
        CGMMSig005Pt09 = CGMM(dm.nnmodel, p_thr=0.9, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
        CGMMSig005Pt09.train()
        CGMMSig001Pt07 = CGMM(dm.nnmodel, p_thr=0.7, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
        CGMMSig001Pt07.train()
        CGMMSig003Pt07 = CGMM(dm.nnmodel, p_thr=0.7, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/33.3, (max(dm.smsz)-min(dm.smsz))/33.3], Gerr = 1.0)
        CGMMSig003Pt07.train()
        CGMMSig005Pt07 = CGMM(dm.nnmodel, p_thr=0.7, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
        CGMMSig005Pt07.train()
        CGMMSig001Pt05 = CGMM(dm.nnmodel, p_thr=0.5, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
        CGMMSig001Pt05.train()
        CGMMSig003Pt05 = CGMM(dm.nnmodel, p_thr=0.5, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/33.3, (max(dm.smsz)-min(dm.smsz))/33.3], Gerr = 1.0)
        CGMMSig003Pt05.train()
        CGMMSig005Pt05 = CGMM(dm.nnmodel, p_thr=0.5, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
        CGMMSig005Pt05.train()
        CGMMSig005Pt03 = CGMM(dm.nnmodel, p_thr=0.3, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
        CGMMSig005Pt03.train()
        CGMMSig010Pt03 = CGMM(dm.nnmodel, p_thr=0.3, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/10, (max(dm.smsz)-min(dm.smsz))/10], Gerr = 1.0)
        CGMMSig010Pt03.train()
        CGMMSig003Pt03 = CGMM(dm.nnmodel, p_thr=0.3, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/33.3, (max(dm.smsz)-min(dm.smsz))/33.3], Gerr = 1.0)
        CGMMSig003Pt03.train()
        
        gmm_name_list = [
                # # (GMM2Sig001, "GMM2Sig001"),
                # # (GMM2Sig002, "GMM2Sig002"),
                # # (GMM2Sig003, "GMM2Sig003"),
                # # (GMM2Sig005, "GMM2Sig005"),
                # (GMM3Sig001, "GMM3Sig001"),
                # # (GMM3Sig002, "GMM3Sig002"),
                # (GMM3Sig003, "GMM3Sig003"),
                # (GMM3Sig005, "GMM3Sig005"),
                
                # (CGMMSig001Pt09, "CGMMSig001Pt09"),
                # (CGMMSig003Pt09, "CGMMSig003Pt09"),
                # (CGMMSig005Pt09, "CGMMSig005Pt09"),
                # (CGMMSig001Pt07, "CGMMSig001Pt07"),
                # (CGMMSig003Pt07, "CGMMSig003Pt07"),
                # (CGMMSig005Pt07, "CGMMSig005Pt07"),
                # (CGMMSig001Pt05, "CGMMSig001Pt05"),
                # (CGMMSig003Pt05, "CGMMSig003Pt05"),
                # (CGMMSig005Pt05, "CGMMSig005Pt05"),
                # (CGMMSig005Pt03, "CGMMSig005Pt03"),
                # (CGMMSig010Pt03, "CGMMSig010Pt03"),
                (CGMMSig003Pt03, "CGMMSig003Pt03"),
        ]
        datotal = setup_datotal(dm, logdir)
        gmmpred = setup_gmmpred(dm, gmm_name_list, logdir)
        gmm_names = [gmm_name for _, gmm_name in gmm_name_list]
        gain_pairs = [
            # (1.0,0.2), 
            # (1.0,0.5), 
            (1.0,1.0), 
            # (1.0,1.2),
            (1.0,1.5),  
            (1.0,2.0),
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
        text = ["<br />".join(["t{}: {}".format(j+1,y_list[i]) for j,y_list in enumerate(y_list_meta) if y_list[i] < -0.5]) for i in range(len(dm.smsz))]
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
