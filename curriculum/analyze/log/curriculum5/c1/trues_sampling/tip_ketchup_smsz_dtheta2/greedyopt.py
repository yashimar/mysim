# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from ......util import *
from .......tasks_domain.util import Rmodel
pd.set_option('display.max_rows', 500)


ONLY_MEAN = "only_mean"
R_EXPEC = "r_expec"
R_EXPEC_LCB = "r_expec_lcb"
SRC_PATH = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/npdata/"


def Help():
    pass


def Run(ct, *args):
    save_sh_dir = "curriculum5/c1/trues_sampling/tip_ketchup_smsz_dthtea2"
    
    smsz = np.linspace(0.03,0.08,100)
    dtheta2 = np.linspace(0.1,1,100)[::-1]
    datotal = {TRUE: None, MEAN: None, SIGMA: None}
    r_types = (TRUE, ONLY_MEAN, R_EXPEC, R_EXPEC_LCB)
    r, policy = dict(), dict()
    for r_type in r_types:
        r[r_type] = np.ones((100,100))*(-1e3)
        policy[r_type] = None
    
    if False:
        sh,esh = get_true_and_bestpolicy_est_state_histories(save_sh_dir, None, None, recreate = False)
        # datotal = np.array(sh["n3ti1"]["da_total"][MEAN]).reshape(100,100) #goのheatmapで反転されるので, np.flipudは必要ない
        datotal[TRUE] = np.flipud(np.array(sh["n3ti1"]["da_total"][MEAN]).reshape(100,100))
        kernel_size = 10
        datotal[MEAN] = cv2.filter2D(datotal[TRUE], -1, np.full((kernel_size,kernel_size),1./(kernel_size**2)))
        datotal[SIGMA] = np.sqrt(cv2.filter2D(datotal[TRUE]**2, -1, np.full((kernel_size,kernel_size),1./(kernel_size**2))) - datotal[MEAN]**2)
        
        for i in range(100):
            for j in range(100):
                print(i, j)
                r[TRUE][i,j] = - 100*max(0, 0.3 - datotal[TRUE][i,j].item())**2 - 20*max(0, datotal[TRUE][i,j].item() - 0.3)**2
                r[ONLY_MEAN][i,j] = - 100*max(0, 0.3 - datotal[MEAN][i,j].item())**2 - 20*max(0, datotal[MEAN][i,j].item() - 0.3)**2
                rdatotal = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[MEAN][i,j].item()], x_var=[0., datotal[SIGMA][i,j].item()**2], with_var=True)
                r[R_EXPEC][i,j] = rdatotal.Y.item()
                r[R_EXPEC_LCB][i,j] = rdatotal.Y.item() - np.sqrt(rdatotal.Var[0,0]).item()
        
        np.save(SRC_PATH+"datotal.npy", datotal[TRUE])
        np.save(SRC_PATH+"datotal_mean.npy", datotal[MEAN])
        np.save(SRC_PATH+"datotal_sigma.npy", datotal[SIGMA])
        np.save(SRC_PATH+"r_true.npy", r[TRUE])
        np.save(SRC_PATH+"r_only_mean.npy", r[ONLY_MEAN])
        np.save(SRC_PATH+"r_expec.npy", r[R_EXPEC])
        np.save(SRC_PATH+"r_expec_lcb.npy", r[R_EXPEC_LCB])
    else:
        datotal[TRUE] = np.load(SRC_PATH+"datotal.npy")
        datotal[MEAN] = np.load(SRC_PATH+"datotal_mean.npy")
        datotal[SIGMA] = np.load(SRC_PATH+"datotal_sigma.npy")
        r[TRUE] = np.load(SRC_PATH+"r_true.npy")
        r[ONLY_MEAN] = np.load(SRC_PATH+"r_only_mean.npy")
        r[R_EXPEC] = np.load(SRC_PATH+"r_expec.npy")
        r[R_EXPEC_LCB] = np.load(SRC_PATH+"r_expec_lcb.npy")
        
    rmatrix = lambda r_type1, r_type2: [r[r_type2][i_policy,i_smsz] for i_policy,i_smsz in zip(np.argmax(r[r_type1], axis=0), range(len(r[TRUE])))]
    for r_type in r_types:
        policy[r_type] = pd.DataFrame({
            "smsz": smsz,
            "best_dtheta2": np.array(dtheta2)[np.argmax(r[r_type], axis=0)],
            "est_return": rmatrix(r_type, r_type),
            "true_return": rmatrix(r_type, TRUE)
        })
        
    colors = ["blue", "orange", "red", "green"]
    fig = go.Figure()
    fig.update_layout(hoverdistance = 5)
    fig.add_trace(go.Heatmap(
        z = datotal[TRUE],
        x = smsz, y = dtheta2,
        colorscale = "Greys",
        # zmin = 0., zmax = 0.55,
        colorbar=dict(
            title="datotal (生データ)",
            titleside="top", ticks="outside",
            x = 1.1,
        ),
    ))
    addpos = {0: (0,-0.0025), 1: (0.0001,-0.0025), 2: (0, 0.0025), 3: (0.0001, 0.0025)}
    for i, r_type in enumerate(r_types):
        fig.add_trace(
            go.Scatter(
                x=[x+addpos[i][0] for x in policy[r_type]["smsz"]], y=[y+addpos[i][1] for y in policy[r_type]["best_dtheta2"]],
                mode='markers', 
                name=r_type,
                marker_color=colors[i],
                # showlegend=False,
            )
        )
    fig.show()
        
    fig = go.Figure()
    for i, r_type in enumerate(r_types):
        fig.add_trace(
            go.Scatter(
                x=policy[r_type]["smsz"], y=[y+0.02*i for y in policy[r_type]["true_return"]],
                mode='markers', 
                name=r_type+" [observed]",
                marker_color=colors[i],
                # showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=policy[r_type]["smsz"], y=policy[r_type]["est_return"],
                mode='lines', 
                name=r_type+" [est]",
                marker_color=colors[i],
                # showlegend=False,
            )
        )
    fig['layout']['yaxis']['range'] = (-8,0.1)
    visc_title_pair = [
        ([True]*len(r_types), "all"),
        ([False,False,True,True], "r_expec & r_expec_lcb")
    ] + [([False]*i+[True]+[False]*(len(r_types)-i-1), r_type) for i,r_type in enumerate(r_types)]
    fig['layout']['updatemenus'] = [{
            "type": "dropdown",
            "buttons": [{
                'label': vis_title,
                'method': "update",
                'args':[
                    {'visible': sum([[c,c] for c in visc],[])},
                    # {'title': ["[{}] {} / {}".format(title,x,y) for x,y,_,_ in xy_limit_pairs for _,title in vis_df_title_pair]}
                ]
            } for i, (visc, vis_title) in enumerate(visc_title_pair)],
            "active": 0,
            # "x": 1.05,
            # "xanchor": "left",
            # "y": 1,
            # "yanchor": "top",
        }]
    fig.show()