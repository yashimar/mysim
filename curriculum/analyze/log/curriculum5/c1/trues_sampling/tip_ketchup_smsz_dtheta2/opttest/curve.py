# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .setup import *


def Run(ct, *args):
    pref_rname_list = [
        # ("Er", "Er"),
        # ("Er_LCB2", "Er_LCB2"),
        # ("GMM4Sig003_gnnsd1_ggmm1", "gmm_gnnsd1.0_ggmm1.0~Er"),
        # ("GMM4Sig003_gnnsd1_ggmm1_LCB2", "gmm_gnnsd1.0_ggmm1.0~Er_LCB2"),
        # ("GMM4Sig003_gnnsd1_ggmm2", "gmm_gnnsd1.0_ggmm2.0~Er"),
        # # ("GMM4Sig003_gnnsd1_ggmm2_LCB2", "gmm_gnnsd1.0_ggmm2.0~Er_LCB2"),
        # ("GMM4Sig005_gnnsd1_ggmm1", "gmm_gnnsd1.0_ggmm1.0~Er"),
        # ("GMM4Sig005_gnnsd1_ggmm1_LCB2", "gmm_gnnsd1.0_ggmm1.0~Er_LCB2"),
        # ("GMM4Sig003_gnnsd1_ggmm3", "gmm_gnnsd1.0_ggmm3.0~Er"),
        # ("GMM4Sig005_gnnsd1_ggmm2", "gmm_gnnsd1.0_ggmm2.0~Er"),
        ("GMM5Sig003_gnnsd1_ggmm1", "gmm_gnnsd1.0_ggmm1.0~Er"),
    ]
    # trial_list = ["t{}".format(i) for i in range(1,2)]
    trial_list = ["t1"]
    
    
    for pref, rname in pref_rname_list:
        save_img_dir = PICTURE_DIR + "opttest/onpolicy/{}/{}/".format(pref, trial_list[0])
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
            gmm = gmm[gmm_name_list[0][1]]
            er = reward[rname]
        #     gmm_concat.append(gmm[gmm_name_list[0][1]])
        #     er_concat.append(reward[rname])
        # datotal = setup_datotal(dm, logdir)
        # gmm = np.mean(gmm_concat, axis=0)
        # er = np.mean(er_concat, axis=0)
        
        
        diffcs = [
            [0, "rgb(0, 0, 0)"],
            [0.01, "rgb(255, 255, 200)"],
            [1, "rgb(255, 0, 0)"],
        ]
        jpx_idx = [[idx_of_the_nearest(dm.dtheta2, x[0]), idx_of_the_nearest(dm.smsz, x[1])] for x in np.array(dm.gmm.jumppoints["X"])]
        jpx_tr = [dm.datotal[RFUNC][idx[0],idx[1]] for idx in jpx_idx]
        jpx_er = [er[idx[0],idx[1]] for idx in jpx_idx]
        jpx_gmm = [gmm[idx[0],idx[1]] for idx in jpx_idx]
        jpy = [y[0] for y in dm.gmm.jumppoints["Y"]]
        linex = [[x,x] for x in np.array(dm.gmm.jumppoints["X"])[:,1]]
        liney = [[y,y] for y in np.array(dm.gmm.jumppoints["X"])[:,0]]
        linetr = [[a,b] for a, b in zip(jpx_tr, jpx_er)]
        linegmm =[[a,b] for a, b in zip(jpy, jpx_gmm)]
        
        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=["評価関数", "飛び値予測", "評価関数 (z軸反転)", "飛び値予測"],
            horizontal_spacing = 0.05,
            specs = [[{"type": "surface"}, {"type": "surface"}],
                     [{"type": "surface"}, {"type": "surface"}]],
        )
        fig.update_layout(
            height=2200, width=2000, 
            # margin=dict(t=100,b=150),
            hoverdistance = 2,
        )
        for i, (z, z_name, cs, sz, lz) in enumerate([
            (er, "er", (-3, 0, "Viridis"), jpx_tr, linetr), 
            (gmm, "gmm", (0, 0.2, diffcs), jpy, linegmm)
        ]):
            for j in range(0,2):
                fig.add_trace(go.Surface(
                    z = z, x = dm.smsz, y = dm.dtheta2,
                    cmin = cs[0], cmax = cs[1], colorscale = cs[2],
                    colorbar = dict(
                        len = 0.15,
                        x = 0.5*(i+1), y = 0.12*(5*j+1),
                    ),
                    showlegend = False,
                ), row=j+1, col=i+1)
                fig.add_trace(go.Scatter3d(
                    z = sz, x = np.array(dm.gmm.jumppoints["X"])[:,1], y = np.array(dm.gmm.jumppoints["X"])[:,0],
                    mode = "markers",
                    showlegend = False,
                    marker = dict(
                        color = "red",
                        size = 4,
                    )
                ), j+1, i+1)
                for tz,tx,ty in zip(lz, linex, liney):
                    fig.add_trace(go.Scatter3d(
                        z = tz, x = tx, y = ty,
                        mode = "lines",
                        line = dict(
                            color = "red",
                        ),
                        showlegend = False,
                    ), j+1, i+1)
                fig['layout']['scene{}'.format(i+2*j+1)]['xaxis']['title'] = "size_srcmouth" 
                fig['layout']['scene{}'.format(i+2*j+1)]['yaxis']['title'] = "dtheta2" 
                fig['layout']['scene{}'.format(i+2*j+1)]['zaxis']['title'] = "evaluation" 
                if j == 1 and i == 0:
                    fig['layout']['scene{}'.format(i+2*j+1)]['zaxis_autorange'] = 'reversed'
        check_or_create_dir(save_img_dir)
        plotly.offline.plot(fig, filename = save_img_dir + "curve.html", auto_open=False)

    