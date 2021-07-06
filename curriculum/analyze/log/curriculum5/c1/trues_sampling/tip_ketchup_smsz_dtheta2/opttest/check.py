# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .learn import *


K10MEAN = "k10_mean"
K10ERR = "k10_error"
NNMEAN = "NN_mean"
NNERR = "NN_error"
NNSD = "NN_sd"
BASE_DIR = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/"

def Run(ct, *args):
    name = "Er/t0.5"
    
    logdir = BASE_DIR + "opttest/logs/{}/".format(name)
    dm = Domain.load(logdir+"dm.pickle")
    if False:
        with open(logdir+"datotal.pickle", mode="rb") as f:
            datotal = pickle.load(f)
    else:
        nnmean, nnerr, nnsd = np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100))
        for idx_dtheta2 in range(len(dm.dtheta2)):
            for idx_smsz in range(len(dm.smsz)):
                print(idx_dtheta2, idx_smsz)
                x_in = [dm.dtheta2[idx_dtheta2], dm.smsz[idx_smsz]]
                xdatota_for_Forward = dm.nnmodel.model.DataX[0:1]; xdatota_for_Forward[0, 0] = x_in[0]; xdatota_for_Forward[0, 1] = x_in[1] #Chainerのバグに対処するため
                nnmean[idx_dtheta2, idx_smsz] = dm.nnmodel.model.Forward(x_data = xdatota_for_Forward, train = False).data.item() #model.Predict(..., x_var=zero).Yと同じ
                nnerr[idx_dtheta2, idx_smsz] = dm.nnmodel.model.ForwardErr(x_data = xdatota_for_Forward, train = False).data.item()
                nnsd[idx_dtheta2, idx_smsz] = np.sqrt(dm.nnmodel.model.Predict(x = x_in, with_var = True).Var[0,0].item())
        datotal = {
            TRUE: dm.datotal[TRUE],
            K10MEAN: np.load(BASE_DIR+"npdata/datotal_mean.npy"),
            K10ERR: np.abs(dm.datotal[TRUE] - np.load(BASE_DIR+"npdata/datotal_mean.npy")),
            NNMEAN: nnmean,
            NNERR: nnerr,
            NNSD: nnsd,
        }
        with open(logdir+"datotal.pickle", mode="wb") as f:
            pickle.dump(datotal, f)
        
    fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=["datotal 生データ (100×100)", "", "10×10平均", "|生データ - 10×10平均|", "NN平均予測", "NN誤差予測"],
        horizontal_spacing = 0.1,
        vertical_spacing = 0.1,
    )
    fig.update_layout(
        height=1600, width=1750, 
        margin=dict(t=100,b=150),
        hoverdistance = 5,
    )
    true_nnerr = [np.abs(v - dm.nnmodel.model.Predict([dtheta2,smsz]).Y.item()) for dtheta2, smsz, v in zip(dm.log["est_opt_dtheta2"], dm.log["smsz"], dm.log["true_datotal"])]
    z_rc_pos_scale_scatterz_scatterscale_set = (
        (datotal[TRUE], 1, 1, 0.46, 0.87, 0., 0.55, None, None, None),
        (datotal[K10MEAN], 2, 1, 0.46, 0.51, 0., 0.55, None, None, None), (datotal[K10ERR], 2, 2, 1.0075, 0.51, 0., 0.3, None, None, None),
        (datotal[NNMEAN], 3, 1, 0.46, 0.15, 0., 0.55, dm.log["true_datotal"], 0., 0.55), (datotal[NNERR], 3, 2, 1.0075, 0.15, 0., 0.3, true_nnerr, 0., 0.3),
    )
    for z, row, col, posx, posy, zmin, zmax, scz, sczmin, sczmax in z_rc_pos_scale_scatterz_scatterscale_set:
        fig.add_trace(go.Heatmap(
            z = z, x = dm.smsz, y = dm.dtheta2,
            colorscale = "Viridis",
            zmin = zmin, zmax = zmax,
            colorbar=dict(
                titleside="top", ticks="outside",
                x = posx, y = posy,
                thickness=23, len = 0.3,
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
                colorscale="Viridis",
                cmin = sczmin,
                cmax = sczmax,
                line = dict(
                    color = "black",
                    width = 1,
                )
            ),
        ), row, col)
    for i in range(1,6):
        fig['layout']['xaxis'+str(i)]['title'] = "size_srcmouth"
        fig['layout']['yaxis'+str(i)]['title'] = "dtheta2"
    fig.show()