# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .setup import *


def Run(ct, *args):
    name = "t0.1/5000/t1"
    if len(args) == 1: name = args[0]
    # save_img_dir = PICTURE_DIR + "opttest/{}/".format(name.replace("/","_"))
    save_img_dir = PICTURE_DIR + "opttest/{}/".format(name)
    
    c_heatmap = False
    c_datotal = True
    # c_obsr = False
    # c_reward_obsr = False
    # c_reward_unobs = False
    c_reward_gmm = True
    # c_reward_gmm_obsr = False
    c_reward_opt = False
    
    logdir = BASE_DIR + "opttest/logs/{}/".format(name)
    dm = Domain.load(logdir+"dm.pickle")
    observations = np.array([
        dm.log["est_opt_dtheta2"],
        dm.log["smsz"]
    ]).T
    GMM2Sig001 = GMM2(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
    GMM2Sig001.train()
    GMM2Sig002 = GMM2(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50], Gerr = 1.0)
    GMM2Sig002.train()
    GMM2Sig003 = GMM2(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/33.3, (max(dm.smsz)-min(dm.smsz))/33.3], Gerr = 1.0)
    GMM2Sig003.train()
    GMM2Sig005 = GMM2(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
    GMM2Sig005.train()
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
    CGMMSig001Pt07 = CGMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
    CGMMSig001Pt07.train(p_thr=0.7)
    CGMMSig003Pt07 = CGMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/33.3, (max(dm.smsz)-min(dm.smsz))/33.3], Gerr = 1.0)
    CGMMSig003Pt07.train(p_thr=0.7)
    CGMMSig005Pt07 = CGMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
    CGMMSig005Pt07.train(p_thr=0.7)
    CGMMSig001Pt05 = CGMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
    CGMMSig001Pt05.train(p_thr=0.5)
    CGMMSig003Pt05 = CGMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/33.3, (max(dm.smsz)-min(dm.smsz))/33.3], Gerr = 1.0)
    CGMMSig003Pt05.train(p_thr=0.5)
    CGMMSig005Pt05 = CGMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
    CGMMSig005Pt05.train(p_thr=0.5)
    gmm_name_list = [
            # (GMM2Sig001, "GMM2Sig001"),
            # (GMM2Sig002, "GMM2Sig002"),
            # (GMM2Sig003, "GMM2Sig003"),
            # (GMM2Sig005, "GMM2Sig005"),
            (GMM3Sig001, "GMM3Sig001"),
            # (GMM3Sig002, "GMM3Sig002"),
            (GMM3Sig003, "GMM3Sig003"),
            (GMM3Sig005, "GMM3Sig005"),
            
            (CGMMSig001Pt09, "CGMMSig001Pt09"),
            (CGMMSig003Pt09, "CGMMSig003Pt09"),
            (CGMMSig005Pt09, "CGMMSig005Pt09"),
            (CGMMSig001Pt07, "CGMMSig001Pt07"),
            (CGMMSig003Pt07, "CGMMSig003Pt07"),
            (CGMMSig005Pt07, "CGMMSig005Pt07"),
            (CGMMSig001Pt05, "CGMMSig001Pt05"),
            (CGMMSig003Pt05, "CGMMSig003Pt05"),
            (CGMMSig005Pt05, "CGMMSig005Pt05"),
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

    datotal[JP1DIFF], datotal[JP2DIFF] = np.ones((100,100))*(-100), np.ones((100,100))*(-100) 
    for idx_dtheta2 in range(len(dm.dtheta2)):
        for idx_smsz in range(len(dm.smsz)):
            datotal[JP1DIFF][idx_dtheta2, idx_smsz] = max(
                datotal[NNMEAN][idx_dtheta2, idx_smsz]-1*datotal[NNERR][idx_dtheta2, idx_smsz]-datotal[TRUE][idx_dtheta2, idx_smsz],
                datotal[TRUE][idx_dtheta2, idx_smsz]-(datotal[NNMEAN][idx_dtheta2, idx_smsz]+1*datotal[NNERR][idx_dtheta2, idx_smsz]),
                0
            )
            datotal[JP2DIFF][idx_dtheta2, idx_smsz] = max(
                datotal[NNMEAN][idx_dtheta2, idx_smsz]-2*datotal[NNERR][idx_dtheta2, idx_smsz]-datotal[TRUE][idx_dtheta2, idx_smsz],
                datotal[TRUE][idx_dtheta2, idx_smsz]-(datotal[NNMEAN][idx_dtheta2, idx_smsz]+2*datotal[NNERR][idx_dtheta2, idx_smsz]),
                0
            )
    
    
    #ヒートマップ
    if c_heatmap:
        print("ヒートマップ")
        nnmean = [datotal[NNMEAN][idx_of_the_nearest(dm.dtheta2, dtheta2), idx_of_the_nearest(dm.smsz, smsz)] for dtheta2, smsz in zip(dm.log["est_opt_dtheta2"], dm.log["smsz"])]
        nnerr = [datotal[NNERR][idx_of_the_nearest(dm.dtheta2, dtheta2), idx_of_the_nearest(dm.smsz, smsz)] for dtheta2, smsz in zip(dm.log["est_opt_dtheta2"], dm.log["smsz"])]
        true_nnerr = [np.abs(v - _nnmean) for _nnmean, v in zip(nnmean, dm.log["true_datotal"])]
        jp1diff = [max(_nnmean-1*_nnerr-v, v-(_nnmean+1*_nnerr), 0) for _nnmean, _nnerr, v in zip(nnmean, nnerr, dm.log["true_datotal"])]
        jp2diff = [max(_nnmean-2*_nnerr-v, v-(_nnmean+2*_nnerr), 0) for _nnmean, _nnerr, v in zip(nnmean, nnerr, dm.log["true_datotal"])]
        n_row = 12
        clength = 0.05
        fig = make_subplots(
            rows=n_row, cols=2, 
            subplot_titles=["datotal 生データ (100×100)", "報酬 生データ (100×100)", 
                            "10×10平均", "|生データ - 10×10平均|", 
                            "NN平均予測", "NN誤差予測", 
                            "NN平均予測", "NN誤差予測", 
                            "|NN平均+/-NN誤差からのずれ| ({}/{})".format(len([x for x in jp1diff if x != 0]),len(dm.log["true_datotal"])), "|NN平均+/-2NN誤差からのずれ| ({}/{})".format(len([x for x in jp2diff if x != 0]),len(dm.log["true_datotal"])), 
                            "|NN平均+/-NN誤差からのずれ| (観測点のみ) ({}/{})".format(len([x for x in jp1diff if x != 0]),len(dm.log["true_datotal"])), "|NN平均+/-2NN誤差からのずれ| (観測点のみ) ({}/{})".format(len([x for x in jp2diff if x != 0]),len(dm.log["true_datotal"])),
                            # "GMM2 sum|NN平均+/-標準偏差 - 飛び値| 1%標準偏差", "GMM2 |NN平均+/-標準偏差 - 飛び値| 2%標準偏差", 
                            # "GMM2 sum|NN平均+/-標準偏差 - 飛び値| 3%標準偏差","",
                            # "GMM2 sum|NN平均+/-標準偏差 - 飛び値| 1%標準偏差", "GMM2 sum|NN平均+/-標準偏差 - 飛び値| 3%標準偏差", 
                            # "GMM2 sum|NN平均+/-標準偏差 - 飛び値| 5%標準偏差",
                            "GMM3 max|NN平均+/-標準偏差 - 飛び値| 1%標準偏差", "CGMM max|NN平均+/-標準偏差 - 飛び値| 1%標準偏差 閾値90%",
                            "GMM3 max|NN平均+/-標準偏差 - 飛び値| 3%標準偏差", "CGMM max|NN平均+/-標準偏差 - 飛び値| 3%標準偏差 閾値90%",
                            "GMM3 max|NN平均+/-標準偏差 - 飛び値| 5%標準偏差", "CGMM max|NN平均+/-標準偏差 - 飛び値| 5%標準偏差 閾値90%",
                            "CGMM max|NN平均+/-標準偏差 - 飛び値| 1%標準偏差 閾値70%", "CGMM max|NN平均+/-標準偏差 - 飛び値| 1%標準偏差 閾値50%",
                            "CGMM max|NN平均+/-標準偏差 - 飛び値| 3%標準偏差 閾値70%", "CGMM max|NN平均+/-標準偏差 - 飛び値| 3%標準偏差 閾値50%",
                            "CGMM max|NN平均+/-標準偏差 - 飛び値| 5%標準偏差 閾値70%", "CGMM max|NN平均+/-標準偏差 - 飛び値| 5%標準偏差 閾値50%",
                            ],
            horizontal_spacing = 0.1,
            vertical_spacing = 0.01,
        )
        fig.update_layout(
            height=600*n_row, width=1750, 
            margin=dict(t=100,b=150),
            hoverdistance = 2,
        )
        diffcs = [
            [0, "rgb(255, 255, 255)"],
            [0.01, "rgb(255, 255, 200)"],
            [1, "rgb(255, 0, 0)"],
        ]
        z_rc_pos_scale_cs_scatterz_scatterscale_set = (
            (datotal[TRUE], 1, 1, 0.46, 0.94, 0., 0.55, None, None, None, None), (dm.datotal[RFUNC], 1, 2, 0.46, 0.94, -3, 0., None, None, None, None),
            (datotal[K10MEAN], 2, 1, 0.46, 0.73, 0., 0.55, None, None, None, None), (datotal[K10ERR], 2, 2, 1.0075, 0.73, 0., 0.3, None, None, None, None),
            (datotal[NNMEAN], 3, 1, 0.46, 0.5, 0., 0.55, None, dm.log["true_datotal"], 0., 0.55), (datotal[NNERR], 3, 2, 1.0075, 0.5, 0., 0.36, None, true_nnerr, 0., 0.3),
            (datotal[NNMEAN], 4, 1, 0.46, 0.5, 0., 0.55, None, None, None, None), (datotal[NNERR], 4, 2, 1.0075, 0.5, 0., 0.36, None, None, None, None),
            (datotal[JP1DIFF], 5, 1, 0.46, 0.28, 0., 0.2, diffcs, jp1diff, 0., 0.2), (datotal[JP2DIFF], 5, 2, 1.0075, 0.28, 0., 0.2, diffcs, jp2diff, 0., 0.2),
            (np.zeros((100,100)), 6, 1, 0.46, 0.06, None, None, diffcs, jp1diff, 0., 0.2), (np.zeros((100,100)), 6, 2, 1.0075, 0.06, None, None, diffcs, jp2diff, 0., 0.2),
            # (gmmpred["GMM2Sig001"], 7, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), (gmmpred["GMM2Sig002"], 7, 2, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), 
            # (gmmpred["GMM2Sig003"], 8, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None),
            # (gmmpred["GMM2Sig001"], 9, 1, 0.46, 0.28, 0., 0.2, diffcs, jp1diff, 0., 0.2), (gmmpred["GMM2Sig002"], 9, 2, 0.46, 0.28, 0., 0.2, diffcs, jp1diff, 0., 0.2), 
            # (gmmpred["GMM2Sig003"], 10, 1, 0.46, 0.28, 0., 0.2, diffcs, jp1diff, 0., 0.2), (gmmpred["GMM2Sig005"], 10, 2, 0.46, 0.28, 0., 0.2, diffcs, jp1diff, 0., 0.2), 
            (gmmpred["GMM3Sig001"], 7, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), (gmmpred["CGMMSig001Pt09"], 7, 2, 0.46, 0.28, 0., 0.2, diffcs, None, None, None),
            (gmmpred["GMM3Sig003"], 8, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), (gmmpred["CGMMSig003Pt09"], 8, 2, 0.46, 0.28, 0., 0.2, diffcs, None, None, None),
            (gmmpred["GMM3Sig005"], 9, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), (gmmpred["CGMMSig005Pt09"], 9, 2, 0.46, 0.28, 0., 0.2, diffcs, None, None, None),
            (gmmpred["CGMMSig001Pt07"], 10, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), (gmmpred["CGMMSig001Pt05"], 10, 2, 0.46, 0.28, 0., 0.2, diffcs, None, None, None),
            (gmmpred["CGMMSig003Pt07"], 11, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), (gmmpred["CGMMSig003Pt05"], 11, 2, 0.46, 0.28, 0., 0.2, diffcs, None, None, None),
            (gmmpred["CGMMSig005Pt07"], 12, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), (gmmpred["CGMMSig005Pt05"], 12, 2, 0.46, 0.28, 0., 0.2, diffcs, None, None, None),
        )
        posx_set = [0.46, 1.0075]
        posy_set = (lambda x: [0.05 + 0.90/(x-1)*i for i in range(x)][::-1])(n_row)
        for z, row, col, posx, posy, zmin, zmax, cs, scz, sczmin, sczmax in z_rc_pos_scale_cs_scatterz_scatterscale_set:
            if np.sum(z) != 0:
                fig.add_trace(go.Heatmap(
                    z = z, x = dm.smsz, y = dm.dtheta2,
                    colorscale = cs if cs != None else "Viridis",
                    zmin = zmin, zmax = zmax,
                    colorbar=dict(
                        titleside="top", ticks="outside",
                        x = posx_set[col-1], y = posy_set[row-1],
                        thickness=23, len = clength,
                    ),
                ), row, col)
                if scz == None: continue
                fig.add_trace(go.Scatter(
                    x = dm.log["smsz"], y=dm.log["est_opt_dtheta2"],
                    mode='markers',
                    showlegend = False,
                    hoverinfo='text',
                    text = ["zvalue: {}<br />ep: {}<br />smsz: {}<br />dtheta2: {}<br />".format(_scz, _ep, _smsz, _dtheta2) for _ep, _scz, _smsz, _dtheta2 in zip(dm.log["ep"], scz, dm.log["smsz"], dm.log["est_opt_dtheta2"])],
                    marker = dict(
                        size = 8,
                        color = scz,
                        colorscale = cs if cs != None else "Viridis",
                        cmin = sczmin,
                        cmax = sczmax,
                        line = dict(
                            color = "black",
                            width = 1,
                        )
                    ),
                ), row, col)
            else:
                if scz == None: continue
                fig.add_trace(go.Scatter(
                    x = dm.log["smsz"], y=dm.log["est_opt_dtheta2"],
                    mode='markers',
                    showlegend = False,
                    hoverinfo='text',
                    text = ["zvalue: {}<br />ep: {}<br />smsz: {}<br />dtheta2: {}<br />".format(_scz, _ep, _smsz, _dtheta2) for _ep, _scz, _smsz, _dtheta2 in zip(dm.log["ep"], scz, dm.log["smsz"], dm.log["est_opt_dtheta2"])],
                    marker = dict(
                        size = 8,
                        color = scz,
                        cmin = sczmin,
                        cmax = sczmax,
                        line = dict(
                            color = "black",
                            width = 1,
                        ),
                        colorscale = cs if cs != None else "Viridis",
                        colorbar=dict(
                            titleside="top", ticks="outside",
                            x = posx_set[col-1], y = posy_set[row-1],
                            thickness=23, len = clength,
                        ),
                    ),
                ), row, col)
        for i in range(1,len(z_rc_pos_scale_cs_scatterz_scatterscale_set)+1):
            fig['layout']['xaxis'+str(i)]['title'] = "size_srcmouth"
            fig['layout']['yaxis'+str(i)]['title'] = "dtheta2"
        check_or_create_dir(save_img_dir)
        plotly.offline.plot(fig, filename = save_img_dir + "heatmap.html", auto_open=False)
    
    
    # #観測重み報酬曲線
    # if c_obsr:
    #     print("観測重み報酬曲線")
    #     obs_types = [name for _, name in obsr_name_list]
    #     trace = defaultdict(list)
    #     for smsz_idx, smsz in enumerate(dm.smsz):
    #         if smsz in dm.log["smsz"]:
    #             log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == smsz]
    #             trace[0].append(go.Scatter(
    #                 x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=[0 for _datotal in np.array(dm.log["true_datotal"])[log_smsz_idx_list]],
    #                 mode='markers', 
    #                 name="True datotal [TrainingData]",
    #                 marker=dict(color="purple", size=8),
    #                     visible=False,
    #             ))
    #         else:
    #             trace[0].append(go.Scatter(x=[], y=[]))
    #         for i,obs_type in enumerate(obs_types):
    #             trace[1+i].append(go.Scatter(
    #                 x=dm.dtheta2, y=obsrcalcs[obs_type][:,smsz_idx],
    #                 mode='lines', 
    #                 name=obs_type,
    #                 line=dict(color="red"),
    #                 visible=False,
    #             ))
    #     for i in range(len(trace)):
    #         trace[i][0].visible = True 
    #     data = sum([trace[i] for i in range(len(trace))], [])   
    #     steps = []
    #     for smsz_idx, smsz in enumerate(dm.smsz):
    #         for j in range(len(trace)):
    #             trace["vis{}".format(j)] = [False]*len(dm.smsz)
    #             trace["vis{}".format(j)][smsz_idx] = True
    #         step = dict(
    #             method="update",
    #             args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
    #                     {"title": "size_srcmouth: {:.4f}".format(smsz)}],
    #         )
    #         steps.append(step)
    #     sliders = [dict(
    #         active=10,
    #         currentvalue={"prefix": "size_srcmouth: "},
    #         pad={"t": 50},
    #         steps=steps,
    #     )]
    #     fig = go.Figure(data=data)
    #     fig.update_layout(
    #         sliders=sliders
    #     )
    #     fig['layout']['xaxis']['title'] = "dtheta2"
    #     fig['layout']['yaxis']['title'] = "reward"
    #     fig['layout']['yaxis']['range'] = (-8,0.5)
    #     for smsz_idx, smsz in enumerate(dm.smsz):
    #         fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    #     check_or_create_dir(save_img_dir)
    #     plotly.offline.plot(fig, filename = save_img_dir + "obsr.html", auto_open=False)


    # #評価関数曲線 (Observation Reward)
    # if c_reward_obsr:
    #     for _, name in obsr_name_list:
    #         trace = defaultdict(list)
    #         print("評価関数曲線 "+name)
    #         for smsz_idx, smsz in enumerate(dm.smsz):
    #             trace[0].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward["normal"][Er][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 2SD[r]",
    #                 line=dict(color="grey", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=2*reward["normal"][Sr][:,smsz_idx],
    #                     color="grey",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[1].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward["normal"][Er][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 1SD[r]",
    #                 line=dict(color="orange", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=1*reward["normal"][Sr][:,smsz_idx],
    #                     color="orange",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[2].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward["normal"][Er][:,smsz_idx]+obsrcalcs[name][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 2SD[r] ({})".format(name),
    #                 line=dict(color="purple", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=2*reward["normal"][Sr][:,smsz_idx],
    #                     color="purple",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[3].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward["normal"][Er][:,smsz_idx]+obsrcalcs[name][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 1SD[r] ({})".format(name),
    #                 line=dict(color="green", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=1*reward["normal"][Sr][:,smsz_idx],
    #                     color="green",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[4].append(go.Scatter(
    #                 x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TRUE][:,smsz_idx]],
    #                 mode='markers', 
    #                 name="True datotal",
    #                 marker=dict(color="blue"),
    #                 visible=False,
    #             ))
    #             if smsz in dm.log["smsz"]:
    #                 log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == smsz]
    #                 trace[5].append(go.Scatter(
    #                     x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["true_datotal"])[log_smsz_idx_list]],
    #                     mode='markers', 
    #                     name="True datotal [TrainingData]",
    #                     marker=dict(color="purple", size=8),
    #                     visible=False,
    #                 ))
    #             else:
    #                 trace[5].append(go.Scatter(x=[], y=[]))
    #         for i in range(len(trace)):
    #             trace[i][0].visible = True 
    #         data = sum([trace[i] for i in range(len(trace))], [])   
    #         steps = []
    #         for smsz_idx, smsz in enumerate(dm.smsz):
    #             for j in range(len(trace)):
    #                 trace["vis{}".format(j)] = [False]*len(dm.smsz)
    #                 trace["vis{}".format(j)][smsz_idx] = True
    #             step = dict(
    #                 method="update",
    #                 args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
    #                     {"title": "size_srcmouth: {:.4f}".format(smsz)}],
    #             )
    #             steps.append(step)
    #         sliders = [dict(
    #             active=10,
    #             currentvalue={"prefix": "size_srcmouth: "},
    #             pad={"t": 50},
    #             steps=steps,
    #         )]
    #         fig = go.Figure(data=data)
    #         fig.update_layout(
    #             sliders=sliders
    #         )
    #         fig['layout']['xaxis']['title'] = "dtheta2"
    #         fig['layout']['yaxis']['title'] = "return"
    #         fig['layout']['yaxis']['range'] = (-8,0.5)
    #         for smsz_idx, smsz in enumerate(dm.smsz):
    #             fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    #         check_or_create_dir(save_img_dir)
    #         plotly.offline.plot(fig, filename = save_img_dir + "return({}).html".format(name), auto_open=False)
            
            
    # #評価関数曲線 (Unobserved SD)
    # if c_reward_unobs:
    #     for _, name in unobs_name_list:
    #         trace = defaultdict(list)
    #         print("評価関数曲線 "+name)
    #         for smsz_idx, smsz in enumerate(dm.smsz):
    #             trace[0].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward["normal"][Er][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 2SD[r]",
    #                 line=dict(color="grey", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=2*reward["normal"][Sr][:,smsz_idx],
    #                     color="grey",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[1].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward["normal"][Er][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 1SD[r]",
    #                 line=dict(color="orange", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=1*reward["normal"][Sr][:,smsz_idx],
    #                     color="orange",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[2].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward["{}~{}".format(name, Er)][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 2SD[r] ({})".format(name),
    #                 line=dict(color="purple", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=2*reward["{}~{}".format(name, Sr)][:,smsz_idx],
    #                     color="purple",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[3].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward["{}~{}".format(name, Er)][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 1SD[r] ({})".format(name),
    #                 line=dict(color="green", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=1*reward["{}~{}".format(name, Sr)][:,smsz_idx],
    #                     color="green",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[4].append(go.Scatter(
    #                 x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TRUE][:,smsz_idx]],
    #                 mode='markers', 
    #                 name="True datotal",
    #                 marker=dict(color="blue"),
    #                 visible=False,
    #             ))
    #             if smsz in dm.log["smsz"]:
    #                 log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == smsz]
    #                 trace[5].append(go.Scatter(
    #                     x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["true_datotal"])[log_smsz_idx_list]],
    #                     mode='markers', 
    #                     name="True datotal [TrainingData]",
    #                     marker=dict(color="purple", size=8),
    #                     visible=False,
    #                 ))
    #             else:
    #                 trace[5].append(go.Scatter(x=[], y=[]))
    #         for i in range(len(trace)):
    #             trace[i][0].visible = True 
    #         data = sum([trace[i] for i in range(len(trace))], [])   
    #         steps = []
    #         for smsz_idx, smsz in enumerate(dm.smsz):
    #             for j in range(len(trace)):
    #                 trace["vis{}".format(j)] = [False]*len(dm.smsz)
    #                 trace["vis{}".format(j)][smsz_idx] = True
    #             step = dict(
    #                 method="update",
    #                 args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
    #                     {"title": "size_srcmouth: {:.4f}".format(smsz)}],
    #             )
    #             steps.append(step)
    #         sliders = [dict(
    #             active=10,
    #             currentvalue={"prefix": "size_srcmouth: "},
    #             pad={"t": 50},
    #             steps=steps,
    #         )]
    #         fig = go.Figure(data=data)
    #         fig.update_layout(
    #             sliders=sliders
    #         )
    #         fig['layout']['xaxis']['title'] = "dtheta2"
    #         fig['layout']['yaxis']['title'] = "return"
    #         fig['layout']['yaxis']['range'] = (-8,0.5)
    #         for smsz_idx, smsz in enumerate(dm.smsz):
    #             fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    #         check_or_create_dir(save_img_dir)
    #         plotly.offline.plot(fig, filename = save_img_dir + "return_{}.html".format(name), auto_open=False)
            
    
    #datotal曲線
    if c_datotal:
        # for n_unobs in unobssds.keys():
        #     for n_gmm in gmmpred.keys():
        #         print("datotal曲線", n_unobs, n_gmm)
        #         trace = defaultdict(list)
        #         for smsz_idx, smsz in enumerate(dm.smsz):
        #             trace[0].append(go.Scatter(
        #                 x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
        #                 mode='lines', 
        #                 name="NNmean+/-2NNerr",
        #                 line=dict(color="grey", dash="dash"),
        #                 visible=False,
        #                 error_y=dict(
        #                     type="data",
        #                     symmetric=True,
        #                     array=2*datotal[NNERR][:,smsz_idx],
        #                     color="grey",
        #                     thickness=1.5,
        #                     width=3,
        #                 )
        #             ))
        #             trace[1].append(go.Scatter(
        #                 x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
        #                 mode='lines', 
        #                 name="NNmean+/-1NNerr",
        #                 line=dict(color="orange", dash="dash"),
        #                 visible=False,
        #                 error_y=dict(
        #                     type="data",
        #                     symmetric=True,
        #                     array=datotal[NNERR][:,smsz_idx],
        #                     color="orange",
        #                     thickness=1.5,
        #                     width=3,
        #                 )
        #             ))
        #             trace[2].append(go.Scatter(
        #                 x=dm.dtheta2, y=datotal[TRUE][:,smsz_idx],
        #                 mode='markers', 
        #                 name="True datotal",
        #                 marker=dict(color="blue"),
        #                 visible=False,
        #             ))
        #             if smsz in dm.log["smsz"]:
        #                 log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == smsz]
        #                 trace[3].append(go.Scatter(
        #                     x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=np.array(dm.log["true_datotal"])[log_smsz_idx_list],
        #                     mode='markers', 
        #                     name="True datotal [TrainingData]",
        #                     marker=dict(color="purple", size=8),
        #                     visible=False,
        #                 ))
        #             else:
        #                 trace[3].append(go.Scatter(x=[], y=[]))
        #             trace[4].append(go.Scatter(
        #                 x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
        #                 mode='lines', 
        #                 name=n_gmm,
        #                 line=dict(color="red", dash="dashdot"),
        #                 visible=False,
        #                 error_y=dict(
        #                     type="data",
        #                     symmetric=True,
        #                     array=gmmpred[n_gmm][:,smsz_idx],
        #                     color="red",
        #                     thickness=1.5,
        #                     width=3,
        #                 )
        #             ))
        #             trace[5].append(go.Scatter(
        #                 x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
        #                 mode='lines', 
        #                 name=n_unobs,
        #                 line=dict(color="red", dash="dashdot"),
        #                 visible=False,
        #                 error_y=dict(
        #                     type="data",
        #                     symmetric=True,
        #                     array=unobssds[n_unobs][:,smsz_idx],
        #                     color="purple",
        #                     thickness=1.5,
        #                     width=3,
        #                 )
        #             ))
        #         for i in range(len(trace)):
        #             trace[i][0].visible = True 
        #         data = sum([trace[i] for i in range(len(trace))], [])   
        #         steps = []
        #         for smsz_idx, smsz in enumerate(dm.smsz):
        #             for j in range(len(trace)):
        #                 trace["vis{}".format(j)] = [False]*len(dm.smsz)
        #                 trace["vis{}".format(j)][smsz_idx] = True
        #             step = dict(
        #                 method="update",
        #                 args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
        #                     {"title": "size_srcmouth: {:.4f}".format(smsz)}],
        #             )
        #             steps.append(step)
        #         sliders = [dict(
        #             active=10,
        #             currentvalue={"prefix": "size_srcmouth: "},
        #             pad={"t": 50},
        #             steps=steps,
        #         )]
        #         fig = go.Figure(data=data)
        #         fig.update_layout(
        #             sliders=sliders
        #         )
        #         fig['layout']['xaxis']['title'] = "dtheta2"
        #         fig['layout']['yaxis']['title'] = "datotal"
        #         fig['layout']['yaxis']['range'] = (-0.05,0.6)
        #         for smsz_idx, smsz in enumerate(dm.smsz):
        #             fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
        #         check_or_create_dir(save_img_dir)
        #         plotly.offline.plot(fig, filename = save_img_dir + "datotal_{}_{}.html".format(n_unobs, n_gmm), auto_open=False)
                
        # for n_unobs in unobssds.keys():
        #     for n_gmm in gmmpred.keys():
        #         print("datotal曲線加算", n_unobs, n_gmm)
        #         trace = defaultdict(list)
        #         for smsz_idx, smsz in enumerate(dm.smsz):
        #             trace[0].append(go.Scatter(
        #                 x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
        #                 mode='lines', 
        #                 name=n_unobs,
        #                 line=dict(color="purple", dash="dashdot"),
        #                 visible=False,
        #                 error_y=dict(
        #                     type="data",
        #                     symmetric=True,
        #                     array=datotal[NNERR][:,smsz_idx]+gmmpred[n_gmm][:,smsz_idx]+unobssds[n_unobs][:,smsz_idx],
        #                     color="purple",
        #                     thickness=1.5,
        #                     width=3,
        #                 )
        #             ))
        #             trace[1].append(go.Scatter(
        #                 x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
        #                 mode='lines', 
        #                 name=n_gmm,
        #                 line=dict(color="red", dash="dashdot"),
        #                 visible=False,
        #                 error_y=dict(
        #                     type="data",
        #                     symmetric=True,
        #                     array=datotal[NNERR][:,smsz_idx]+gmmpred[n_gmm][:,smsz_idx],
        #                     color="red",
        #                     thickness=1.5,
        #                     width=3,
        #                 )
        #             ))
        #             trace[2].append(go.Scatter(
        #                 x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
        #                 mode='lines', 
        #                 name="NNmean+/-1NNerr",
        #                 line=dict(color="orange", dash="dash"),
        #                 visible=False,
        #                 error_y=dict(
        #                     type="data",
        #                     symmetric=True,
        #                     array=datotal[NNERR][:,smsz_idx],
        #                     color="orange",
        #                     thickness=1.5,
        #                     width=3,
        #                 )
        #             ))
        #             trace[3].append(go.Scatter(
        #                 x=dm.dtheta2, y=datotal[TRUE][:,smsz_idx],
        #                 mode='markers', 
        #                 name="True datotal",
        #                 marker=dict(color="blue"),
        #                 visible=False,
        #             ))
        #             if smsz in dm.log["smsz"]:
        #                 log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == smsz]
        #                 trace[4].append(go.Scatter(
        #                     x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=np.array(dm.log["true_datotal"])[log_smsz_idx_list],
        #                     mode='markers', 
        #                     name="True datotal [TrainingData]",
        #                     marker=dict(color="purple", size=8),
        #                     visible=False,
        #                 ))
        #             else:
        #                 trace[4].append(go.Scatter(x=[], y=[]))
        #         for i in range(len(trace)):
        #             trace[i][0].visible = True 
        #         data = sum([trace[i] for i in range(len(trace))], [])   
        #         steps = []
        #         for smsz_idx, smsz in enumerate(dm.smsz):
        #             for j in range(len(trace)):
        #                 trace["vis{}".format(j)] = [False]*len(dm.smsz)
        #                 trace["vis{}".format(j)][smsz_idx] = True
        #             step = dict(
        #                 method="update",
        #                 args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
        #                     {"title": "size_srcmouth: {:.4f}".format(smsz)}],
        #             )
        #             steps.append(step)
        #         sliders = [dict(
        #             active=10,
        #             currentvalue={"prefix": "size_srcmouth: "},
        #             pad={"t": 50},
        #             steps=steps,
        #         )]
        #         fig = go.Figure(data=data)
        #         fig.update_layout(
        #             sliders=sliders
        #         )
        #         fig['layout']['xaxis']['title'] = "dtheta2"
        #         fig['layout']['yaxis']['title'] = "datotal"
        #         fig['layout']['yaxis']['range'] = (-0.05,0.6)
        #         for smsz_idx, smsz in enumerate(dm.smsz):
        #             fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
        #         check_or_create_dir(save_img_dir)
        #         plotly.offline.plot(fig, filename = save_img_dir + "datotal_addsd_{}_{}.html".format(n_unobs, n_gmm), auto_open=False)
                
        for n_gmm in [
            # "GMM3Sig003",
            # "CGMMSig003Pt09",
            "CGMMSig003Pt07",
            # "CGMMSig003Pt05",
        ]:
            for gnnsd, ggmm in [
                (1.0, 1.0),
                (1.0, 0.5),
            ]:
                print("datotal曲線 gmm加算", n_gmm)
                trace = defaultdict(list)
                for smsz_idx, smsz in enumerate(dm.smsz):
                    trace[0].append(go.Scatter(
                            x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
                            mode='lines', 
                            name=n_gmm,
                            line=dict(color="red", dash="dashdot"),
                            visible=False,
                            error_y=dict(
                                type="data",
                                symmetric=True,
                                array=gnnsd*datotal[NNERR][:,smsz_idx]+ggmm*gmmpred[n_gmm][:,smsz_idx],
                                color="red",
                                thickness=1.5,
                                width=3,
                        )
                    ))
                    trace[1].append(go.Scatter(
                            x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
                            mode='lines', 
                            name="NNmean+/-1NNerr",
                            line=dict(color="orange", dash="dash"),
                            visible=False,
                            error_y=dict(
                                type="data",
                                symmetric=True,
                                array=gnnsd*datotal[NNERR][:,smsz_idx],
                                color="orange",
                                thickness=1.5,
                                width=3,
                            )
                    ))
                    trace[2].append(go.Scatter(
                            x=dm.dtheta2, y=datotal[TRUE][:,smsz_idx],
                            mode='markers', 
                            name="Unobs",
                            marker=dict(color="blue"),
                            visible=False,
                    ))
                    # if smsz in dm.log["smsz"]:
                    #     log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == smsz]
                    #     trace[3].append(go.Scatter(
                    #             x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=np.array(dm.log["true_datotal"])[log_smsz_idx_list],
                    #             mode='markers', 
                    #             name="True datotal [TrainingData]",
                    #             marker=dict(color="purple", size=8),
                    #             visible=False,
                    #     ))
                    # else:
                    #     trace[3].append(go.Scatter(x=[], y=[]))
                    for i,addv in enumerate(range(-5,6)):
                        if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                            tmp_smsz = dm.smsz[smsz_idx+addv]
                            if tmp_smsz in dm.log["smsz"]:
                                log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == tmp_smsz]
                                trace[3+i].append(go.Scatter(
                                        x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=np.array(dm.log["true_datotal"])[log_smsz_idx_list],
                                        mode='markers', 
                                        name="Obs {:.3f}".format(tmp_smsz),
                                        marker=dict(
                                            color= "purple" if addv == 0 else "gray", 
                                            size=8,
                                        ),
                                        visible=False,
                                ))
                            else:
                                trace[3+i].append(go.Scatter(x=[], y=[]))
                        else:
                            trace[3+i].append(go.Scatter(x=[], y=[]))
                for i in range(len(trace)):
                    trace[i][0].visible = True 
                data = sum([trace[i] for i in range(len(trace))], [])   
                steps = []
                for smsz_idx, smsz in enumerate(dm.smsz):
                    for j in range(len(trace)):
                        trace["vis{}".format(j)] = [False]*len(dm.smsz)
                        trace["vis{}".format(j)][smsz_idx] = True
                    step = dict(
                            method="update",
                            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                                {"title": "size_srcmouth: {:.4f}".format(smsz)}],
                    )
                    steps.append(step)
                sliders = [dict(
                        active=10,
                        currentvalue={"prefix": "size_srcmouth: "},
                        pad={"t": 50},
                        steps=steps,
                )]
                fig = go.Figure(data=data)
                fig.update_layout(
                        sliders=sliders
                )
                fig['layout']['xaxis']['title'] = "dtheta2"
                fig['layout']['yaxis']['title'] = "datotal"
                fig['layout']['yaxis']['range'] = (-0.05,0.6)
                for smsz_idx, smsz in enumerate(dm.smsz):
                    fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
                check_or_create_dir(save_img_dir)
                plotly.offline.plot(fig, filename = save_img_dir + "datotal_addsd_{}_gnnsd{}_ggmm{}.html".format(n_gmm, gnnsd, ggmm), auto_open=False)

    
    #評価関数曲線 (GMM)
    if c_reward_gmm:
        for gain_type in [
                "gnnsd1.0_ggmm1.0", 
                "gnnsd1.0_ggmm0.5", 
            ]:
            for name in (
                # lambda t, gain_type=gain_type: "GMM2Sig001_{}~{}".format(gain_type, t), 
                # lambda t, gain_type=gain_type: "GMM3Sig003_{}~{}".format(gain_type, t), 
                # lambda t, gain_type=gain_type: "CGMMSig003Pt09_{}~{}".format(gain_type, t), 
                lambda t, gain_type=gain_type: "CGMMSig003Pt07_{}~{}".format(gain_type, t), 
                # lambda t, gain_type=gain_type: "CGMMSig003Pt05_{}~{}".format(gain_type, t), 
            ):
                trace = defaultdict(list)
                print("評価関数曲線 "+name(Er))
                for smsz_idx, smsz in enumerate(dm.smsz):
                    trace[0].append(go.Scatter(
                        x=dm.dtheta2, y=reward[Er][:,smsz_idx],
                        mode='lines', 
                        name="E[r] - 2SD[r]",
                        line=dict(color="grey", dash="dash"),
                        visible=False,
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=[0]*len(dm.dtheta2),
                            arrayminus=2*reward[Sr][:,smsz_idx],
                            color="grey",
                            thickness=1.5,
                            width=3,
                        )
                    ))
                    trace[1].append(go.Scatter(
                        x=dm.dtheta2, y=reward[Er][:,smsz_idx],
                        mode='lines', 
                        name="E[r] - 1SD[r]",
                        line=dict(color="orange", dash="dash"),
                        visible=False,
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=[0]*len(dm.dtheta2),
                            arrayminus=1*reward[Sr][:,smsz_idx],
                            color="orange",
                            thickness=1.5,
                            width=3,
                        )
                    ))
                    trace[2].append(go.Scatter(
                        x=dm.dtheta2, y=reward[name(Er)][:,smsz_idx],
                        mode='lines', 
                        name="E[r] - 2SD[r] ({})".format(name(Er)),
                        line=dict(color="purple", dash="dash"),
                        visible=False,
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=[0]*len(dm.dtheta2),
                            arrayminus=2*reward[name(Sr)][:,smsz_idx],
                            color="purple",
                            thickness=1.5,
                            width=3,
                        )
                    ))
                    trace[3].append(go.Scatter(
                        x=dm.dtheta2, y=reward[name(Er)][:,smsz_idx],
                        mode='lines', 
                        name="E[r] - 1SD[r] ({})".format(name(Er)),
                        line=dict(color="green", dash="dash"),
                        visible=False,
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=[0]*len(dm.dtheta2),
                            arrayminus=1*reward[name(Sr)][:,smsz_idx],
                            color="green",
                            thickness=1.5,
                            width=3,
                        )
                    ))
                    for i,addv in enumerate(range(0,1)):
                        if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                            tmp_smsz = dm.smsz[smsz_idx+addv]
                            trace[4+i].append(go.Scatter(
                                x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TRUE][:,smsz_idx+addv]],
                                mode='markers', 
                                name="Unobs {:.3f}".format(tmp_smsz),
                                marker=dict(
                                            color= "blue" if addv == 0 else "grey", 
                                            size=8,
                                            symbol="x",
                                        ),
                                visible=False,
                            ))
                        else:
                            trace[4+i].append(go.Scatter(x=[], y=[]))
                    for i,addv in enumerate(range(-4,5)):
                        if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                            tmp_smsz = dm.smsz[smsz_idx+addv]
                            if tmp_smsz in dm.log["smsz"]:
                                log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == tmp_smsz]
                                trace[5+i].append(go.Scatter(
                                        x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["true_datotal"])[log_smsz_idx_list]],
                                        mode='markers', 
                                        name="Obs {:.3f}".format(tmp_smsz),
                                        marker=dict(
                                            color= "purple" if addv == 0 else "pink", 
                                            size=8,
                                        ),
                                        visible=False,
                                ))
                            else:
                                trace[5+i].append(go.Scatter(x=[], y=[]))
                        else:
                            trace[5+i].append(go.Scatter(x=[], y=[]))
                for i in range(len(trace)):
                    trace[i][0].visible = True 
                data = sum([trace[i] for i in range(len(trace))], [])   
                steps = []
                for smsz_idx, smsz in enumerate(dm.smsz):
                    for j in range(len(trace)):
                        trace["vis{}".format(j)] = [False]*len(dm.smsz)
                        trace["vis{}".format(j)][smsz_idx] = True
                    step = dict(
                        method="update",
                        args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                            {"title": "size_srcmouth: {:.4f}".format(smsz)}],
                    )
                    steps.append(step)
                sliders = [dict(
                    active=10,
                    currentvalue={"prefix": "size_srcmouth: "},
                    pad={"t": 50},
                    steps=steps,
                )]
                fig = go.Figure(data=data)
                fig.update_layout(
                    sliders=sliders
                )
                fig['layout']['xaxis']['title'] = "dtheta2"
                fig['layout']['yaxis']['title'] = "return"
                fig['layout']['yaxis']['range'] = (-8,0.5)
                for smsz_idx, smsz in enumerate(dm.smsz):
                    fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
                check_or_create_dir(save_img_dir)
                plotly.offline.plot(fig, filename = save_img_dir + "return_{}_{}.html".format(gain_type, name(Er)), auto_open=False)
            
            
    # #評価関数曲線 (GMM + Unobserved SD)
    # gmm_types = [lambda t, name=name: "{}~{}".format(name, t) for name in gmmpred.keys()]
    # us_types = [name for name in unobssds.keys()]
    # names = [lambda t, us_type=us_type, gmm_type=gmm_type: "{}-{}".format(us_type, gmm_type(t)) for us_type in us_types for gmm_type in gmm_types]
    # if c_reward_gmm_obsr:
    #     for name in names:
    #         trace = defaultdict(list)
    #         print("評価関数曲線 "+name(Er))
    #         for smsz_idx, smsz in enumerate(dm.smsz):
    #             trace[0].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward[Er][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 2SD[r]",
    #                 line=dict(color="grey", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=2*reward[Sr][:,smsz_idx],
    #                     color="grey",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[1].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward[Er][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 1SD[r]",
    #                 line=dict(color="orange", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=1*reward[Sr][:,smsz_idx],
    #                     color="orange",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[2].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward[name(Er)][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 2SD[r] ({})".format(name(Er)),
    #                 line=dict(color="purple", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=2*reward[name(Sr)][:,smsz_idx],
    #                     color="purple",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[3].append(go.Scatter(
    #                 x=dm.dtheta2, y=reward[name(Er)][:,smsz_idx],
    #                 mode='lines', 
    #                 name="E[r] - 1SD[r] ({})".format(name(Er)),
    #                 line=dict(color="green", dash="dash"),
    #                 visible=False,
    #                 error_y=dict(
    #                     type="data",
    #                     symmetric=False,
    #                     array=[0]*len(dm.dtheta2),
    #                     arrayminus=1*reward[name(Sr)][:,smsz_idx],
    #                     color="green",
    #                     thickness=1.5,
    #                     width=3,
    #                 )
    #             ))
    #             trace[4].append(go.Scatter(
    #                 x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TRUE][:,smsz_idx]],
    #                 mode='markers', 
    #                 name="True datotal",
    #                 marker=dict(color="blue"),
    #                 visible=False,
    #             ))
    #             if smsz in dm.log["smsz"]:
    #                 log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == smsz]
    #                 trace[5].append(go.Scatter(
    #                     x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["true_datotal"])[log_smsz_idx_list]],
    #                     mode='markers', 
    #                     name="True datotal [TrainingData]",
    #                     marker=dict(color="purple", size=8),
    #                     visible=False,
    #                 ))
    #             else:
    #                 trace[5].append(go.Scatter(x=[], y=[]))
    #         for i in range(len(trace)):
    #             trace[i][0].visible = True 
    #         data = sum([trace[i] for i in range(len(trace))], [])   
    #         steps = []
    #         for smsz_idx, smsz in enumerate(dm.smsz):
    #             for j in range(len(trace)):
    #                 trace["vis{}".format(j)] = [False]*len(dm.smsz)
    #                 trace["vis{}".format(j)][smsz_idx] = True
    #             step = dict(
    #                 method="update",
    #                 args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
    #                     {"title": "size_srcmouth: {:.4f}".format(smsz)}],
    #             )
    #             steps.append(step)
    #         sliders = [dict(
    #             active=10,
    #             currentvalue={"prefix": "size_srcmouth: "},
    #             pad={"t": 50},
    #             steps=steps,
    #         )]
    #         fig = go.Figure(data=data)
    #         fig.update_layout(
    #             sliders=sliders
    #         )
    #         fig['layout']['xaxis']['title'] = "dtheta2"
    #         fig['layout']['yaxis']['title'] = "return"
    #         fig['layout']['yaxis']['range'] = (-8,0.5)
    #         for smsz_idx, smsz in enumerate(dm.smsz):
    #             fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    #         check_or_create_dir(save_img_dir)
    #         plotly.offline.plot(fig, filename = save_img_dir + "return_{}.html".format(name(Er)), auto_open=False)
    
    
    #最適化された評価関数曲線
    if c_reward_opt:
        print("最適化された評価関数曲線")
        r_types = [Er, "Er_LCB2"]
        gains = ["gnnsd{}_ggmm{}".format(gnnsd, ggmm) for gnnsd, ggmm in gain_pairs]
        gmm_types = deepcopy(gmm_names)
        r_types += ["{}_{}~{}".format(gmm_type, gain, r_type) for r_type in r_types for gmm_type in gmm_types for gain in gains]
        fig = go.Figure()
        for name in r_types:
            opt_dtheta2_list = np.argmax(reward[name], axis = 0)
            fig.add_trace(
                go.Scatter(
                    x=dm.smsz, y=[smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[RFUNC].T, opt_dtheta2_list))],
                    mode='markers', 
                    name="{}".format(name),
                )
            )
            # fig.add_trace(
            #     go.Scatter(
            #         x=dm.smsz, y=[smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(reward[name].T, opt_dtheta2_list))],
            #         mode='lines', 
            #         name=name+" [Est Opt]",
            #     )
            # )
        fig['layout']['yaxis']['range'] = (-8,0.1)
        fig['layout']['xaxis']['title'] = "size_srcmouth"
        fig['layout']['yaxis']['title'] = "return"
        check_or_create_dir(save_img_dir)
        plotly.offline.plot(fig, filename = save_img_dir + "opt_return4.html", auto_open=False)