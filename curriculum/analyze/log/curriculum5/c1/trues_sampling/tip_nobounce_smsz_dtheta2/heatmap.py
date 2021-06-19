# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from ......util import *
from .......tasks_domain.util import Rmodel
import cv2


def Help():
    pass


def Run(ct, *args):
    log_name_list = [
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/0_5g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/5_10g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/10_15g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/15_16g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/16_17g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/17_18g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/18_19g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/19_20g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/20_29g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/29_30g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/30_40g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/40_50g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/50_60g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/60_70g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/70_80g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/80_90g0",
        "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2/90_100g0",
    ]
    save_sh_dir = "curriculum5/c1/trues_sampling/tip_nobounce_smsz_dthtea2"
    
    node_states_dim_pair = [
        ["n0", [("size_srcmouth", 1), ("material2", 4), ("dtheta2", 1), ("p_pour_trg", 2)]],
        ["n2b", [("lp_pour", 3), ]],
        ["n2c", [("skill", 1), ]],
        ["n3ti1", [("da_total", 1),]],
        ["n3ti2", [("lp_flow", 2), ("flow_var", 1)]],
        ["n4ti", [("da_pour", 1), ("da_spill2", 1)]],
        ["n4tir1", [(".r", 1), ]],
        ["n4tir2", [(".r", 1), ]],
        ["n3sa1", [("da_total", 1),]],
        ["n3sa2", [("lp_flow", 2), ("flow_var", 1)]],
        ["n4sa", [("da_pour", 1), ("da_spill2", 1)]],
        ["n4sar1", [(".r", 1), ]],
        ["n4sar2", [(".r", 1), ]],
    ]
    
    sh, esh = get_true_and_bestpolicy_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate=False)
    datotal = np.array(sh["n3ti1"]["da_total"][MEAN]).reshape(100,100) #goのheatmapで反転されるので, np.flipudは必要ない
    kernel_size = 10
    datotal_mean = cv2.filter2D(datotal, -1, np.full((kernel_size,kernel_size),1./(kernel_size**2)))
    datotal_sigma = np.sqrt(cv2.filter2D(datotal**2, -1, np.full((kernel_size,kernel_size),1./(kernel_size**2))) - datotal_mean**2)
    smsz = np.linspace(0.03,0.08,100)
    dtheta2 = np.linspace(0.1,1,100)
    rdatotal_mean = np.zeros((100,100))
    rdatotal_sigma = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            rdatotal = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal_mean[i,j].item()], x_var=[0., datotal_sigma[i,j].item()**2], with_var=True)
            rdatotal_mean[i,j] = rdatotal.Y.item()
            rdatotal_sigma[i,j] = np.sqrt(rdatotal.Var[0,0]).item()
    
    anno_text = [[
        "mean: {}<br />sigma: {}<br />rdatotal_mean: {}<br />rdatotal_sigma: {}<br />true: {}<br />size_srcmouth: {}<br />dtheta2: {}<br />"
    .format(datotal_mean[i,j],datotal_sigma[i,j],rdatotal_mean[i,j],rdatotal_sigma[i,j],datotal[i,j],smsz[j],dtheta2[i]) for j in range(100)] for i in range(100)]
    fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=["da_total ({}×{}平均)".format(kernel_size,kernel_size), "da_total ({}×{}標準偏差)".format(kernel_size,kernel_size), "Rdatotal (平均)", "Rdatotal (標準偏差)","datotal (生データ)"],
        horizontal_spacing = 0.1,
        vertical_spacing = 0.1,
    )
    fig.update_layout(
                height=1600, width=1800, 
                margin=dict(t=100,b=150),
                hoverdistance = 5,
            )
    fig.add_trace(go.Heatmap(
        z = datotal_mean,
        x = smsz, y = dtheta2,
        colorscale = "Viridis",
        zmin = 0., zmax = 0.55,
        colorbar=dict(
            title="datotal (mean)",
            titleside="top", ticks="outside",
            x = 0.46, y = 0.87,
            thickness=23, len = 0.3,
        ),
        hoverinfo='text',
        text = anno_text,
    ), 1, 1)
    fig.add_trace(go.Heatmap(
        z = datotal_sigma,
        x = smsz, y = dtheta2,
        colorbar=dict(
            title="datotal (sigma)",
            titleside="top", ticks="outside",
            x = 1.0075, y = 0.87,
            thickness=23, len = 0.3,
        ),
        hoverinfo='text',
        text = anno_text,
    ), 1, 2)
    fig.add_trace(go.Heatmap(
        z = rdatotal_mean,
        x = smsz, y = dtheta2,
        zmin = -3.0, zmax = 0.,
        colorbar=dict(
            title="Rdatotal (mean)",
            titleside="top", ticks="outside",
            x = 0.46, y = 0.51,
            thickness=23, len = 0.3,
        ),
        hoverinfo='text',
        text = anno_text,
    ), 2, 1)
    fig.add_trace(go.Heatmap(
        z = rdatotal_sigma,
        x = smsz, y = dtheta2,
        colorbar=dict(
            title="Rdatotal (sigma)",
            titleside="top", ticks="outside",
            x = 1.0075, y = 0.51,
            thickness=23, len = 0.3,
        ),
        hoverinfo='text',
        text = anno_text,
    ), 2, 2)
    fig.add_trace(go.Heatmap(
        z = datotal,
        x = smsz, y = dtheta2,
        colorscale = "Viridis",
        zmin = 0., zmax = 0.55,
        colorbar=dict(
            title="datotal",
            titleside="top", ticks="outside",
            x = 0.46, y = 0.15,
            thickness=23, len = 0.3,
        ),
        hoverinfo='text',
        text = anno_text,
    ), 3, 1)
    fig.add_annotation(dict(font=dict(color='black',size=15),
                                                x=0,
                                                y=-0.1,
                                                # showarrow=False,
                                                text="gh_abs: [0.25], material2: nobounce, dtheta1: 0.014",
                                                # textangle=0,
                                                xanchor='left',
                                                xref="paper",
                                                yref="paper"
                                                ))
    for i in range(1,6):
        fig['layout']['xaxis'+str(i)]['title'] = "size_srcmouth"
        fig['layout']['yaxis'+str(i)]['title'] = "dtheta2"
    
    saveimg_dir = PICTURE_DIR + save_sh_dir.replace("/", "_") + "/"
    check_or_create_dir(saveimg_dir)
    plotly.offline.plot(fig, filename = saveimg_dir + "heatmap{}.html".format(kernel_size), auto_open=False)