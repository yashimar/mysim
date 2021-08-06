# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .setup import *
from .learn2 import *


def Run(ct, *args):
    name = "GMM6Sig0002_LCB1/t1"
    if len(args) == 1: name = args[0]
    
    save_img_dir = PICTURE_DIR + "opttest/onpolicy/{}/".format(name)        
    logdir = BASE_DIR + "opttest/logs/onpolicy/{}/".format(name)
    if "Er" in name:
        dm = Domain.load(logdir+"dm.pickle")
        gmm = GMM6(dm.nnmodel, diag_sigma=[(1.0-0.1)/33.3, (0.8-0.3)/33.3], Gerr = 1.0)
        gmm.train(dm.log["true_r_at_est_opt_dthtea2"])
        reward = setup_reward(dm, logdir)
    else:
        dm = Domain2.load(logdir+"dm.pickle")
        setup_datotal(dm, logdir)
        reward = setup_reward2(dm, logdir)
        gmm = dm.gmm
        gmm.train(dm.log["true_r_at_est_opt_dthtea2"])
    X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ])
    gr = gmm.predict(X).reshape(100,100)
    er = reward[Er]
    sr = reward[Sr]
    ersr = er - sr
    ev = er - 1*(sr + gr)
        
    jpx_idx = [[idx_of_the_nearest(dm.dtheta2, x[0]), idx_of_the_nearest(dm.smsz, x[1])] for x in np.array(gmm.jumppoints["X"])]
    jpx_tr = [dm.datotal[RFUNC][idx[0],idx[1]] for idx in jpx_idx]
    jpx_er = np.array([er[idx[0],idx[1]] for idx in jpx_idx])
    jpx_ersr = np.array([ersr[idx[0],idx[1]] for idx in jpx_idx])
    jpx_ev = np.array([ev[idx[0],idx[1]] for idx in jpx_idx])
    jpx_gr = np.array([gr[idx[0],idx[1]] for idx in jpx_idx])
    jpy = [y for y in gmm.jumppoints["Y"]]
    linex = [[x,x] for x in np.array(gmm.jumppoints["X"])[:,1]]
    liney = [[y,y] for y in np.array(gmm.jumppoints["X"])[:,0]]
    lineer = [[a,b] for a, b in zip(jpx_tr, jpx_er)]
    lineersr = [[a,b] for a, b in zip(jpx_tr, jpx_ersr)]
    lineev = [[a,b] for a, b in zip(jpx_tr, jpx_ev)]
    linegr = [[a,b] for a, b in zip(jpy, jpx_gr)]
    
    
    posx_set = [0.46, 1.0075]
    posy_set = (lambda x: [0.1 + 0.8/(x-1)*i for i in range(x)][::-1])(4)
    clen = 0.08
    rcs = [
        [0, "rgb(100, 100, 255)"],
        [0.3, "rgb(130, 120, 150)"],
        [0.5, "rgb(170, 130, 100)"],
        [0.7, "rgb(200, 140, 50)"],
        [0.95, "rgb(255, 150, 0)"],
        [1, "rgb(0, 255, 0)"],
    ]
    diffcs = [
        [0, "rgb(0, 0, 0)"],
        [0.01, "rgb(255, 255, 200)"],
        [1, "rgb(255, 0, 0)"],
    ]
    fig = make_subplots(
        rows=4, cols=2, 
        subplot_titles=["E[r]", "E[r] - SD[r]", "E[r] - SD[r] - 評価関数飛び値予測", "評価関数飛び値予測", "E[r] (z軸反転)", "E[r] - SD[r] (z軸反転)", "E[r] - SD[r] - 評価関数飛び値予測 (z軸反転)", "評価関数飛び値予測"],
        horizontal_spacing = 0.0,
        vertical_spacing = 0.02,
        specs = [[{"type": "surface"}, {"type": "surface"}],
                 [{"type": "surface"}, {"type": "surface"}],
                 [{"type": "surface"}, {"type": "surface"}],
                 [{"type": "surface"}, {"type": "surface"}]],
    )
    fig.update_layout(
        height=3600, width=2000, 
        # margin=dict(t=100,b=150),
        hoverdistance = 2,
    )
    fig.add_trace(go.Surface(
        z = er, x = dm.smsz, y = dm.dtheta2,
        cmin = -8, cmax = 0, colorscale = "Viridis",
        colorbar = dict(
            len = clen,
            x = posx_set[0], y = posy_set[0],
        ),
        showlegend = False,
    ), 1,1)
    fig.add_trace(go.Scatter3d(
        z = jpx_tr, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
        marker = dict(
            color = "red",
            size = 4,
        )
    ), 1, 1)
    for tz,tx,ty in zip(lineer, linex, liney):
        fig.add_trace(go.Scatter3d(
             z = tz, x = tx, y = ty,
             mode = "lines",
             line = dict(
                 color = "red",
             ),
            showlegend = False,
        ), 1, 1)
    fig.add_trace(go.Surface(
        z = ersr, x = dm.smsz, y = dm.dtheta2,
        cmin = -8, cmax = 0, colorscale = "Viridis",
        colorbar = dict(
            len = clen,
            x = posx_set[1], y = posy_set[0],
        ),
        showlegend = False,
    ), 1,2)
    fig.add_trace(go.Scatter3d(
        z = jpx_tr, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
        marker = dict(
            color = "red",
            size = 4,
        )
    ), 1, 2)
    for tz,tx,ty in zip(lineersr, linex, liney):
        fig.add_trace(go.Scatter3d(
             z = tz, x = tx, y = ty,
             mode = "lines",
             line = dict(
                 color = "red",
             ),
            showlegend = False,
        ), 1, 2)
    fig.add_trace(go.Surface(
        z = ev, x = dm.smsz, y = dm.dtheta2,
        cmin = -8, cmax = 0, colorscale = "Viridis",
        colorbar = dict(
            len = clen,
            x = posx_set[0], y = posy_set[1],
        ),
        showlegend = False,
    ), 2,1)
    fig.add_trace(go.Scatter3d(
        z = jpx_tr, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
        marker = dict(
            color = "red",
            size = 4,
        )
    ), 2, 1)
    for tz,tx,ty in zip(lineev, linex, liney):
        fig.add_trace(go.Scatter3d(
             z = tz, x = tx, y = ty,
             mode = "lines",
             line = dict(
                 color = "red",
             ),
            showlegend = False,
        ), 2, 1)
    fig.add_trace(go.Surface(
        z = gr, x = dm.smsz, y = dm.dtheta2,
        cmin = 0, cmax = 6, colorscale = diffcs,
        colorbar = dict(
            len = clen,
            x = posx_set[1], y = posy_set[1],
        ),
        showlegend = False,
    ), 2,2)
    fig.add_trace(go.Scatter3d(
        z = jpy, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
        marker = dict(
            color = "red",
            size = 4,
        )
    ), 2, 2)
    for tz,tx,ty in zip(linegr, linex, liney):
        fig.add_trace(go.Scatter3d(
             z = tz, x = tx, y = ty,
             mode = "lines",
             line = dict(
                 color = "red",
             ),
            showlegend = False,
        ), 2, 2)
    fig.add_trace(go.Surface(
        z = er, x = dm.smsz, y = dm.dtheta2,
        cmin = -8, cmax = 0, colorscale = "Viridis",
        colorbar = dict(
            len = clen,
            x = posx_set[0], y = posy_set[2],
        ),
        showlegend = False,
    ), 3,1)
    fig.add_trace(go.Scatter3d(
        z = jpx_tr, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
        marker = dict(
            color = "red",
            size = 4,
        )
    ), 3, 1)
    for tz,tx,ty in zip(lineer, linex, liney):
        fig.add_trace(go.Scatter3d(
             z = tz, x = tx, y = ty,
             mode = "lines",
             line = dict(
                 color = "red",
             ),
            showlegend = False,
        ), 3, 1)
    fig.add_trace(go.Surface(
        z = ersr, x = dm.smsz, y = dm.dtheta2,
        cmin = -8, cmax = 0, colorscale = "Viridis",
        colorbar = dict(
            len = clen,
            x = posx_set[1], y = posy_set[2],
        ),
        showlegend = False,
    ), 3,2)
    fig.add_trace(go.Scatter3d(
        z = jpx_tr, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
        marker = dict(
            color = "red",
            size = 4,
        )
    ), 3, 2)
    for tz,tx,ty in zip(lineersr, linex, liney):
        fig.add_trace(go.Scatter3d(
             z = tz, x = tx, y = ty,
             mode = "lines",
             line = dict(
                 color = "red",
             ),
            showlegend = False,
        ), 3, 2)
    fig.add_trace(go.Surface(
        z = ev, x = dm.smsz, y = dm.dtheta2,
        cmin = -8, cmax = 0, colorscale = "Viridis",
        colorbar = dict(
            len = clen,
            x = posx_set[0], y = posy_set[3],
        ),
        showlegend = False,
    ), 4,1)
    fig.add_trace(go.Scatter3d(
        z = jpx_tr, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
        marker = dict(
            color = "red",
            size = 4,
        )
    ), 4, 1)
    for tz,tx,ty in zip(lineev, linex, liney):
        fig.add_trace(go.Scatter3d(
             z = tz, x = tx, y = ty,
             mode = "lines",
             line = dict(
                 color = "red",
             ),
            showlegend = False,
        ), 4, 1)
    fig.add_trace(go.Surface(
        z = gr, x = dm.smsz, y = dm.dtheta2,
        cmin = 0, cmax = 6, colorscale = diffcs,
        colorbar = dict(
            len = clen,
            x = posx_set[1], y = posy_set[3],
        ),
        showlegend = False,
    ), 4,2)
    fig.add_trace(go.Scatter3d(
        z = jpy, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
        marker = dict(
            color = "red",
            size = 4,
        )
    ), 4, 2)
    for tz,tx,ty in zip(linegr, linex, liney):
        fig.add_trace(go.Scatter3d(
             z = tz, x = tx, y = ty,
             mode = "lines",
             line = dict(
                 color = "red",
             ),
            showlegend = False,
        ), 4, 2)
    for j in range(0,4):
        for i in range(0,2):
            fig['layout']['scene{}'.format(i+2*j+1)]['xaxis']['title'] = "size_srcmouth" 
            fig['layout']['scene{}'.format(i+2*j+1)]['yaxis']['title'] = "dtheta2" 
            fig['layout']['scene{}'.format(i+2*j+1)]['zaxis']['title'] = "evaluation" if not (i%2==1 and j%2==1) else "estimation"
            if j==2 or (j==3 and i==0):
                fig['layout']['scene{}'.format(i+2*j+1)]['zaxis_autorange'] = 'reversed'
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "curve.html", auto_open=False)

    