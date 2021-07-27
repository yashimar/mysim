# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .setup import *


def Run(ct, *args):
    pref_rname_list = [
        ("Er", "Er"),
        ("Er_LCB2", "Er_LCB2"),
        # ("GMM4Sig003_gnnsd1_ggmm1", "gmm_gnnsd1.0_ggmm1.0~Er"),
        # ("GMM4Sig003_gnnsd1_ggmm1_LCB2", "gmm_gnnsd1.0_ggmm1.0~Er_LCB2"),
        # ("GMM4Sig003_gnnsd1_ggmm2", "gmm_gnnsd1.0_ggmm2.0~Er"),
        # # ("GMM4Sig003_gnnsd1_ggmm2_LCB2", "gmm_gnnsd1.0_ggmm2.0~Er_LCB2"),
        # ("GMM4Sig005_gnnsd1_ggmm1", "gmm_gnnsd1.0_ggmm1.0~Er"),
        # ("GMM4Sig005_gnnsd1_ggmm1_LCB2", "gmm_gnnsd1.0_ggmm1.0~Er_LCB2"),
        # ("GMM4Sig003_gnnsd1_ggmm3", "gmm_gnnsd1.0_ggmm3.0~Er"),
        # ("GMM4Sig005_gnnsd1_ggmm2", "gmm_gnnsd1.0_ggmm2.0~Er"),
    ]
    trial_list = ["t{}".format(i) for i in range(2,30)]
    
    
    for pref, rname in pref_rname_list:
        save_img_dir = PICTURE_DIR + "opttest/onpolicy/{}/".format(pref)
        name_list = ["onpolicy/{}/{}".format(pref,trial) for trial in trial_list]
        
        gmm_concat = []
        er_concat = []
        for name in name_list:
            logdir = BASE_DIR + "opttest/logs/{}/".format(name)
            dm = Domain.load(logdir+"dm.pickle")

            setup_datotal(dm, logdir)
            gmm_name_list = [None if dm.gmm == None else (dm.gmm, "gmm")]
            gmm = setup_gmmpred(dm, gmm_name_list, logdir)
            setup_unobssd(dm, [], logdir)
            gmm_names = [gmm_name for _, gmm_name in gmm_name_list]
            try:
                gp = dm.gain_pairs
            except: 
                gp = (1.0 ,1.0)
            gain_pairs = [gp]
            reward = setup_reward(dm, logdir, gmm_names, gain_pairs)
            gmm_concat.append(gmm[gmm_name_list[0][1]])
            er_concat.append(reward[rname])
        datotal = setup_datotal(dm, logdir)
        rc = dm.datotal[RFUNC].reshape(100*100)
        idx = [i for i,r in enumerate(rc) if r<-0.7]
        rc = rc[idx]
        smsz = np.array([smsz for smsz in dm.smsz]*100)[idx]
        dtheta2 = np.array(sum([[dtheta2]*100 for dtheta2 in dm.dtheta2],[]))[idx]
        
        gmm = np.mean(gmm_concat, axis=0)
        er = np.mean(er_concat, axis=0)
    
        n_row = 3
        clength = 0.2
        fig = make_subplots(
            rows=n_row, cols=2, 
            subplot_titles=["datotal 生データ (100×100)", "報酬 生データ (100×100)", 
                            "飛び値モデル (真値報酬-0.7以下の地点プロット)", "評価関数 (真値報酬-0.7以下の地点プロット)", 
                            "飛び値モデル", "評価関数", 
                            ],
            horizontal_spacing = 0.1,
            vertical_spacing = 0.05,
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
            (gmm, 2, 1, 0.46, 0.28, 0., 0.05, diffcs, "badr", -3, 0), (er, 2, 2, 0.46, 0.94, -3, 0., None, "badr", -3, 0),
            (gmm, 3, 1, 0.46, 0.28, 0., 0.05, diffcs, None, None, None), (er, 3, 2, 0.46, 0.94, -3, 0., None, None, None, None),
        )
        posx_set = [0.46, 1.0075]
        posy_set = (lambda x: [0.1 + 0.7/(x-1)*i for i in range(x)][::-1])(n_row)
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
                if scz != "badr": continue
                fig.add_trace(go.Scatter(
                    x = smsz, y = dtheta2,
                    mode='markers',
                    showlegend = False,
                    # hoverinfo='text',
                    # text = ["zvalue: {}<br />ep: {}<br />smsz: {}<br />dtheta2: {}<br />".format(_scz, _ep, _smsz, _dtheta2) for _ep, _scz, _smsz, _dtheta2 in zip(dm.log["ep"], scz, dm.log["smsz"], dm.log["est_opt_dtheta2"])],
                    marker = dict(
                        size = 4,
                        color = rc,
                        colorscale = "Viridis",
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
        # fig.show()
    