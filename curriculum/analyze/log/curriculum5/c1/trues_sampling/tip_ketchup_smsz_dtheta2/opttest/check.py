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
JP1 = "|JumpPoint(G=1) - NNmean|"
JP2 = "|JumpPoint(G=2) - NNmean|"
JP1DIFF = "|JumpPoint - (NNmean+/-1NNerr)|"
JP2DIFF = "|JumpPoint - (NNmean+/-2NNerr)|"
Er = "Er"
Sr = "Sr"
Er_1LCB = "Er_1LCB"
Er_2LCB = "Er_2LCB"
ErJP1 = "ErJP1"
SrJP1 = "SrJP1"
ErJP1_1LCB = "ErJP1_1LCB"
ErJP1_2LCB = "ErJP1_2LCB"
ErJP2 = "ErJP2"
SrJP2 = "SrJP2"
ErJP2_1LCB = "ErJP2_1LCB"
ErJP2_2LCB = "ErJP2_2LCB"
BASE_DIR = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/"


def Run(ct, *args):
    name = "Er/t0.1_fixed"
    save_img_dir = PICTURE_DIR + "opttest/{}/".format(name.replace("/","_"))
    
    logdir = BASE_DIR + "opttest/logs/{}/".format(name)
    dm = Domain.load(logdir+"dm.pickle")
    gmm1 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50], Gerr = 1.0)
    gmm1.train()
    gmm2 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50], Gerr = 2.0)
    gmm2.train()
    if True:
        with open(logdir+"datotal.pickle", mode="rb") as f:
            datotal = pickle.load(f)
        with open(logdir+"reward.pickle", mode="rb") as f:
            reward = pickle.load(f)
    else:
        nnmean, nnerr, nnsd, gmmjp1, gmmjp2 = np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100))
        er, sr, er_1LCB, er_2LCB, erJP1, srJP1, erJP1_1LCB, erJP1_2LCB, erJP2, srJP2, erJP2_1LCB, erJP2_2LCB = np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100))
        for idx_dtheta2, dtheta2 in enumerate(dm.dtheta2):
            for idx_smsz, smsz in enumerate(dm.smsz):
                print(idx_dtheta2, idx_smsz)
                x_in = [dtheta2, smsz]
                xdatota_for_Forward = dm.nnmodel.model.DataX[0:1]; xdatota_for_Forward[0, 0] = x_in[0]; xdatota_for_Forward[0, 1] = x_in[1] #Chainerのバグに対処するため
                nnmean[idx_dtheta2, idx_smsz] = dm.nnmodel.model.Forward(x_data = xdatota_for_Forward, train = False).data.item() #model.Predict(..., x_var=zero).Yと同じ
                nnerr[idx_dtheta2, idx_smsz] = dm.nnmodel.model.ForwardErr(x_data = xdatota_for_Forward, train = False).data.item()
                nnsd[idx_dtheta2, idx_smsz] = np.sqrt(dm.nnmodel.model.Predict(x = x_in, with_var = True).Var[0,0].item())
                gmmjp1[idx_dtheta2, idx_smsz] = gmm1.predict([dtheta2, smsz]).item()
                gmmjp2[idx_dtheta2, idx_smsz] = gmm2.predict([dtheta2, smsz]).item()
                
                r = Rmodel("Fdatotal_gentle").Predict(x=[0.3, nnmean[idx_dtheta2, idx_smsz]], x_var=[0, nnsd[idx_dtheta2, idx_smsz]**2], with_var=True)
                rmean, rsd = r.Y.item(), np.sqrt(r.Var[0,0]).item()
                er[idx_dtheta2, idx_smsz] = rmean
                sr[idx_dtheta2, idx_smsz] = rsd
                er_1LCB[idx_dtheta2, idx_smsz] = rmean - 1*rsd
                er_2LCB[idx_dtheta2, idx_smsz] = rmean - 2*rsd
                
                rjp1 = Rmodel("Fdatotal_gentle").Predict(x=[0.3, nnmean[idx_dtheta2, idx_smsz]], x_var=[0, max(nnsd[idx_dtheta2, idx_smsz], gmmjp1[idx_dtheta2, idx_smsz])**2], with_var=True)
                rjp1mean, rjp1sd = rjp1.Y.item(), np.sqrt(rjp1.Var[0,0]).item()
                erJP1[idx_dtheta2, idx_smsz] = rjp1mean
                srJP1[idx_dtheta2, idx_smsz] = rjp1sd
                erJP1_1LCB[idx_dtheta2, idx_smsz] = rjp1mean - 1*rjp1sd
                erJP1_2LCB[idx_dtheta2, idx_smsz] = rjp1mean - 2*rjp1sd
                
                rjp2 = Rmodel("Fdatotal_gentle").Predict(x=[0.3, nnmean[idx_dtheta2, idx_smsz]], x_var=[0, max(nnsd[idx_dtheta2, idx_smsz], gmmjp2[idx_dtheta2, idx_smsz])**2], with_var=True)
                rjp2mean, rjp2sd = rjp2.Y.item(), np.sqrt(rjp2.Var[0,0]).item()
                erJP2[idx_dtheta2, idx_smsz] = rjp2mean
                srJP2[idx_dtheta2, idx_smsz] = rjp2sd
                erJP2_1LCB[idx_dtheta2, idx_smsz] = rjp2mean - 1*rjp2sd
                erJP2_2LCB[idx_dtheta2, idx_smsz] = rjp2mean - 2*rjp2sd
        datotal = {
            TRUE: dm.datotal[TRUE],
            K10MEAN: np.load(BASE_DIR+"npdata/datotal_mean.npy"),
            K10ERR: np.abs(dm.datotal[TRUE] - np.load(BASE_DIR+"npdata/datotal_mean.npy")),
            NNMEAN: nnmean,
            NNERR: nnerr,
            NNSD: nnsd,
            JP1: gmmjp1,
            JP2: gmmjp2,
        }
        reward = {
            Er: er,
            Sr: sr,
            Er_1LCB: er_1LCB,
            Er_2LCB: er_2LCB,
            ErJP1: erJP1,
            SrJP1: srJP1, 
            ErJP1_1LCB: erJP1_1LCB,
            ErJP1_2LCB: erJP1_2LCB,
            ErJP2: erJP2,
            SrJP2: srJP2, 
            ErJP2_1LCB: erJP2_1LCB,
            ErJP2_2LCB: erJP2_2LCB,
        }
        with open(logdir+"datotal.pickle", mode="wb") as f:
            pickle.dump(datotal, f)
        with open(logdir+"reward.pickle", mode="wb") as f:
            pickle.dump(reward, f)
        
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
    print("ヒートマップ")
    fig = make_subplots(
        rows=4, cols=2, 
        subplot_titles=["datotal 生データ (100×100)", "", "10×10平均", "|生データ - 10×10平均|", "NN平均予測", "NN誤差予測", "|NN平均+/-NN誤差からのずれ|", "|NN平均+/-2NN誤差からのずれ|"],
        horizontal_spacing = 0.1,
        vertical_spacing = 0.1,
    )
    fig.update_layout(
        height=2200, width=1750, 
        margin=dict(t=100,b=150),
        hoverdistance = 5,
    )
    nnmean = [datotal[NNMEAN][idx_of_the_nearest(dm.dtheta2, dtheta2), idx_of_the_nearest(dm.smsz, smsz)] for dtheta2, smsz in zip(dm.log["est_opt_dtheta2"], dm.log["smsz"])]
    nnerr = [datotal[NNERR][idx_of_the_nearest(dm.dtheta2, dtheta2), idx_of_the_nearest(dm.smsz, smsz)] for dtheta2, smsz in zip(dm.log["est_opt_dtheta2"], dm.log["smsz"])]
    true_nnerr = [np.abs(v - _nnmean) for _nnmean, v in zip(nnmean, dm.log["true_datotal"])]
    jp1diff = [max(_nnmean-1*_nnerr-v, v-(_nnmean+1*_nnerr), 0) for _nnmean, _nnerr, v in zip(nnmean, nnerr, dm.log["true_datotal"])]
    jp2diff = [max(_nnmean-2*_nnerr-v, v-(_nnmean+2*_nnerr), 0) for _nnmean, _nnerr, v in zip(nnmean, nnerr, dm.log["true_datotal"])]
    z_rc_pos_scale_scatterz_scatterscale_set = (
        (datotal[TRUE], 1, 1, 0.46, 0.91, 0., 0.55, None, None, None),
        (datotal[K10MEAN], 2, 1, 0.46, 0.63, 0., 0.55, None, None, None), (datotal[K10ERR], 2, 2, 1.0075, 0.63, 0., 0.3, None, None, None),
        (datotal[NNMEAN], 3, 1, 0.46, 0.36, 0., 0.55, dm.log["true_datotal"], 0., 0.55), (datotal[NNERR], 3, 2, 1.0075, 0.36, 0., 0.36, true_nnerr, 0., 0.3),
        (datotal[JP1DIFF], 4, 1, 0.46, 0.08, 0., 0.2, jp1diff, 0., 0.2), (datotal[JP2DIFF], 4, 2, 1.0075, 0.08, 0., 0.2, jp2diff, 0., 0.2),
    )
    for z, row, col, posx, posy, zmin, zmax, scz, sczmin, sczmax in z_rc_pos_scale_scatterz_scatterscale_set:
        fig.add_trace(go.Heatmap(
            z = z, x = dm.smsz, y = dm.dtheta2,
            colorscale = "Viridis",
            zmin = zmin, zmax = zmax,
            colorbar=dict(
                titleside="top", ticks="outside",
                x = posx, y = posy,
                thickness=23, len = 0.19,
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
    for i in range(1,len(z_rc_pos_scale_scatterz_scatterscale_set)+2):
        fig['layout']['xaxis'+str(i)]['title'] = "size_srcmouth"
        fig['layout']['yaxis'+str(i)]['title'] = "dtheta2"
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "heatmap.html", auto_open=False)
    
    
    #datotal曲線
    for JP, ErJP, SrJP in [(JP1, ErJP1, SrJP1), (JP2, ErJP2, SrJP2)]:
        print("datotal曲線 "+JP)
        trace = defaultdict(list)
        for smsz_idx, smsz in enumerate(dm.smsz):
            trace[0].append(go.Scatter(
                x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
                mode='lines', 
                name="NNmean+/-2NNerr",
                line=dict(color="grey", dash="dash"),
                visible=False,
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=2*datotal[NNERR][:,smsz_idx],
                    color="grey",
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
                    array=datotal[NNERR][:,smsz_idx],
                    color="orange",
                    thickness=1.5,
                    width=3,
                )
            ))
            trace[2].append(go.Scatter(
                x=dm.dtheta2, y=datotal[TRUE][:,smsz_idx],
                mode='markers', 
                name="True datotal",
                marker=dict(color="blue"),
                visible=False,
            ))
            if smsz in dm.log["smsz"]:
                log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == smsz]
                trace[3].append(go.Scatter(
                    x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=np.array(dm.log["true_datotal"])[log_smsz_idx_list],
                    mode='markers', 
                    name="True datotal [TrainingData]",
                    marker=dict(color="purple", size=8),
                    visible=False,
                ))
            else:
                trace[3].append(go.Scatter(x=[], y=[]))
            trace[4].append(go.Scatter(
                x=dm.dtheta2, y=datotal[NNMEAN][:,smsz_idx],
                mode='lines', 
                name=JP,
                line=dict(color="red", dash="dashdot"),
                visible=False,
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=datotal[JP][:,smsz_idx],
                    color="red",
                    thickness=1.5,
                    width=3,
                )
            ))
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
        plotly.offline.plot(fig, filename = save_img_dir + "curve({}).html".format(JP), auto_open=False)

    
    #評価関数曲線
    for JP, ErJP, SrJP in [(JP1, ErJP1, SrJP1), (JP2, ErJP2, SrJP2)]:
        trace = defaultdict(list)
        print("評価関数曲線 "+JP)
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
                x=dm.dtheta2, y=reward[ErJP][:,smsz_idx],
                mode='lines', 
                name="E[r] - 2SD[r] ({})".format(JP),
                line=dict(color="purple", dash="dash"),
                visible=False,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[0]*len(dm.dtheta2),
                    arrayminus=2*reward[SrJP][:,smsz_idx],
                    color="purple",
                    thickness=1.5,
                    width=3,
                )
            ))
            trace[3].append(go.Scatter(
                x=dm.dtheta2, y=reward[ErJP][:,smsz_idx],
                mode='lines', 
                name="E[r] - 1SD[r] ({})".format(JP),
                line=dict(color="green", dash="dash"),
                visible=False,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[0]*len(dm.dtheta2),
                    arrayminus=1*reward[SrJP][:,smsz_idx],
                    color="green",
                    thickness=1.5,
                    width=3,
                )
            ))
            trace[4].append(go.Scatter(
                x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TRUE][:,smsz_idx]],
                mode='markers', 
                name="True datotal",
                marker=dict(color="blue"),
                visible=False,
            ))
            if smsz in dm.log["smsz"]:
                log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, log_smsz in enumerate(dm.log["smsz"]) if log_smsz == smsz]
                trace[5].append(go.Scatter(
                    x=np.array(dm.log["est_opt_dtheta2"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["true_datotal"])[log_smsz_idx_list]],
                    mode='markers', 
                    name="True datotal [TrainingData]",
                    marker=dict(color="purple", size=8),
                    visible=False,
                ))
            else:
                trace[5].append(go.Scatter(x=[], y=[]))
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
        plotly.offline.plot(fig, filename = save_img_dir + "return({}).html".format(JP), auto_open=False)
    
    
    #最適化された評価関数曲線
    print("最適化された評価関数曲線")
    r_types = (Er, Er_1LCB, Er_2LCB, ErJP1, ErJP1_1LCB, ErJP1_2LCB)
    fig = go.Figure()
    for r_type in r_types:
        opt_dtheta2_list = np.argmax(reward[r_type], axis = 0)
        fig.add_trace(
            go.Scatter(
                x=dm.smsz, y=[smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[RFUNC].T, opt_dtheta2_list))],
                mode='markers', 
                name=r_type+" [True Value]",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dm.smsz, y=[smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(reward[r_type].T, opt_dtheta2_list))],
                mode='lines', 
                name=r_type+" [Est Opt]",
            )
        )
    fig['layout']['yaxis']['range'] = (-8,0.1)
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "return"
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "opt_return.html", auto_open=False)