# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from ......util import *
from .......tasks_domain.util import Rmodel
from scipy.ndimage import minimum_filter, maximum_filter, median_filter, percentile_filter, gaussian_filter
pd.set_option('display.max_rows', 500)

SRC_PATH = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/npdata/"

MAX = "max"
MIN = "min"
MIN_OUTLIER = "min_outlier"
MAX_OUTLIER = "max_outlier"
MEDIAN_DTHETAWISE = "median_dthetawise"
MAX_DTHETAWISE = "max_dthetawise"
MIN_DTHETAWISE = "min_dthetawise"
MIN_OUTLIER_DTHETAWISE = "min_outlier_dtehatwise"
MAX_OUTLIER_DTHETAWISE = "max_outlier_dtehatwise"
MEDIAN = "median"
ONLY_MEAN = "only_mean"
R_TRUE = "r_true"
R_EXPEC = "r_expec"
R_STD = "r_std"
R_EXPEC_LCB1 = "r_expec_lcb1"
R_EXPEC_LCB2 = "r_expec_lcb2"
R_EXPEC_LCB4 = "r_expec_lcb4"
R_EXPEC_LCB8 = "r_expec_lcb8"
# R_EXPEC_MINOUTLIER = "r_expec_min_outlier"
# R_EXPEC_MAXOUTLIER = "r_expec_max_outlier"
# R_EXPEC_LCB_MINOUTLIER = "r_expec_lcb_min_outlier"
# R_EXPEC_OUTLIER = "r_expec_outlier"
P_OUTLIER = "p_outlier"
P_OUTLIER_DTHETAWISE = "p_outlier_dthetawise"
P_MIN = "p_min"
P_MAX = "p_max"
P_MINMAX = "p_minmax"
P_MINMAX_DIFF = "p_minmax_diff"
P_MINMAX_SUM = "p_minmax_sum"
P_1090_DIFF = "p_1090_diff"
P_0595_DIFF = "p_0595_diff"
P_1090_SUM = "p_1090_sum"
P_0595_SUM = "p_0595_sum"
P_MIN_DTHETAWISE = "p_min_dthetawise"
P_MAX_DTHETAWISE = "p_max_dthetawise"
P_MINMAX_DTHETAWISE = "p_minmax_dthetawise"
# R_EXPEC_LCB_OUTLIER = "r_expec_lcb_outlier"
# R_EXPEC_SUMOUTLIER = "r_expec_sum_outlier"
# R_EXPEC_LCB_SUMOUTLIER = "r_expec_lcb_sum_outlier"


def Help():
    pass


def rfunc(v):
    return - 100*max(0, 0.3 - v)**2 - 20*max(0, v - 0.3)**2


def moving_percentile(p,v_min,v_max):
    return v_min + p*(v_max - v_min)


def Run(ct, *args):
    ms_kernel_size = 5
    filter_kernel_size = 3
    save_sh_dir = "curriculum5/c1/trues_sampling/tip_ketchup_smsz_dthtea2"
    save_dir = PICTURE_DIR + save_sh_dir.replace("/","_") + "/greedy_opt/{}_{}/".format(ms_kernel_size, filter_kernel_size)
    # SRC_PATH = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/npdata/"
    src_path = SRC_PATH + "{}_{}/".format(ms_kernel_size, filter_kernel_size)
    
    smsz = np.linspace(0.03,0.08,100)
    dtheta2 = np.linspace(0.1,1,100)[::-1]
    datotal = {TRUE: None, MEAN: None, SIGMA: None}
    r_types = [R_TRUE, 
               ONLY_MEAN, 
               R_EXPEC, 
               R_STD,
               "05",
               "95",
               "10",
               "90",
               P_MIN,
               P_MAX,
               P_MINMAX,
               P_MINMAX_DIFF,
               P_MINMAX_SUM,
               P_0595_DIFF,
               P_1090_SUM,
               P_0595_SUM,
               P_MIN_DTHETAWISE,
               P_MAX_DTHETAWISE,
               P_MINMAX_DTHETAWISE,
            #    R_EXPEC_LCB1,
            #    R_EXPEC_LCB2,
            #    R_EXPEC_LCB4,
            #    R_EXPEC_LCB8,
            #    P_OUTLIER,
            P_OUTLIER_DTHETAWISE,
            #    R_EXPEC_MINOUTLIER, R_EXPEC_LCB_MINOUTLIER, R_EXPEC_OUTLIER, R_EXPEC_MAXOUTLIER, R_EXPEC_SUMOUTLIER, R_EXPEC_LCB_OUTLIER, R_EXPEC_LCB_SUMOUTLIER
    ]
    r, policy = dict(), dict()
    for r_type in r_types:
        r[r_type] = np.ones((100,100))*(-1e3)
        policy[r_type] = None
    
    if False:
        sh,esh = get_true_and_bestpolicy_est_state_histories(save_sh_dir, None, None, recreate = False)
        # datotal = np.array(sh["n3ti1"]["da_total"][MEAN]).reshape(100,100) #goのheatmapで反転されるので, np.flipudは必要ない
        datotal[TRUE] = np.flipud(np.array(sh["n3ti1"]["da_total"][MEAN]).reshape(100,100))
        datotal[MEAN] = cv2.filter2D(datotal[TRUE], -1, np.full((ms_kernel_size,ms_kernel_size),1./(ms_kernel_size**2)))
        datotal[SIGMA] = np.sqrt(cv2.filter2D(datotal[TRUE]**2, -1, np.full((ms_kernel_size,ms_kernel_size),1./(ms_kernel_size**2))) - datotal[MEAN]**2)
        datotal[MIN_OUTLIER] = median_filter(datotal[TRUE], size=filter_kernel_size) - minimum_filter(datotal[TRUE], size=filter_kernel_size)
        datotal[MAX_OUTLIER] = maximum_filter(datotal[TRUE], size=filter_kernel_size) - median_filter(datotal[TRUE], size=filter_kernel_size)
        datotal[MEDIAN] = median_filter(datotal[TRUE], size=filter_kernel_size)
        datotal[MEDIAN_DTHETAWISE] = np.array([median_filter(row,size=filter_kernel_size) for row in datotal[TRUE].T]).T
        datotal[MIN_DTHETAWISE] = np.array([minimum_filter(row,size=filter_kernel_size) for row in datotal[TRUE].T]).T
        datotal[MAX_DTHETAWISE] = np.array([maximum_filter(row,size=filter_kernel_size) for row in datotal[TRUE].T]).T
        # datotal[MIN_OUTLIER_DTHETAWISE] = datotal[MEDIAN_DTHETAWISE] - datotal[MIN_DTHETAWISE]
        # datotal[MAX_OUTLIER_DTHETAWISE] = datotal[MAX_DTHETAWISE] - datotal[MEDIAN_DTHETAWISE]
        datotal[MIN] = minimum_filter(datotal[TRUE], size=filter_kernel_size)
        datotal[MAX] = maximum_filter(datotal[TRUE], size=filter_kernel_size)
        datotal["95"] = percentile_filter(datotal[TRUE], 95, size=filter_kernel_size)
        datotal["05"] = percentile_filter(datotal[TRUE], 5, size=filter_kernel_size)
        datotal["90"] = percentile_filter(datotal[TRUE], 90, size=filter_kernel_size)
        datotal["10"] = percentile_filter(datotal[TRUE], 10, size=filter_kernel_size)
        
        for i in range(100):
            for j in range(100):
                print(i, j)
                r[R_TRUE][i,j] = rfunc(datotal[TRUE][i,j].item())
                r[ONLY_MEAN][i,j] = rfunc(datotal[MEAN][i,j].item())
                rdatotal = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[MEAN][i,j].item()], x_var=[0., datotal[SIGMA][i,j].item()**2], with_var=True)
                r[R_EXPEC][i,j] = rdatotal.Y.item()
                r[R_STD][i,j] = np.sqrt(rdatotal.Var[0,0]).item()
                # r[R_EXPEC_LCB1][i,j] = r[R_EXPEC][i,j] - 1*r[R_STD][i,j]
                # r[R_EXPEC_LCB2][i,j] = r[R_EXPEC][i,j] - 2*r[R_STD][i,j]
                # r[R_EXPEC_LCB4][i,j] = r[R_EXPEC][i,j] - 4*r[R_STD][i,j]
                # r[R_EXPEC_LCB8][i,j] = r[R_EXPEC][i,j] - 8*r[R_STD][i,j]
                # rdatotal_minoutlier = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[MEAN][i,j].item()], x_var=[0., datotal[SIGMA][i,j].item()**2], with_var=True)
                # r[R_EXPEC_MINOUTLIER][i,j] = min(rdatotal.Y.item(), rfunc(datotal[MEAN][i,j].item() - datotal[MIN_OUTLIER][i,j].item()))
                # r[R_EXPEC_MAXOUTLIER][i,j] = min(rdatotal.Y.item(), rfunc(datotal[MEAN][i,j].item() + datotal[MAX_OUTLIER][i,j].item()))
                # r[R_EXPEC_LCB_MINOUTLIER][i,j] = min(rdatotal.Y.item(), rfunc(datotal[MEAN][i,j].item() - datotal[MIN_OUTLIER][i,j].item())) - np.sqrt(rdatotal.Var[0,0]).item()
                # r[R_EXPEC_OUTLIER][i,j] = min(r[R_EXPEC][i,j], rfunc(datotal[MEAN][i,j] - datotal[MIN_OUTLIER][i,j]), rfunc(datotal[MEAN][i,j] + datotal[MAX_OUTLIER][i,j]))               
                # r[R_EXPEC_SUMOUTLIER][i,j] = rfunc(datotal[MEAN][i,j].item() - datotal[MIN_OUTLIER][i,j].item()) + rfunc(datotal[MEAN][i,j].item() + datotal[MAX_OUTLIER][i,j].item())
                # r[R_EXPEC_LCB_OUTLIER][i,j] = min(rdatotal.Y.item(), rfunc(datotal[MEAN][i,j].item() - datotal[MIN_OUTLIER][i,j].item()), rfunc(datotal[MEAN][i,j].item() + datotal[MAX_OUTLIER][i,j].item())) - np.sqrt(rdatotal.Var[0,0]).item()
                # r[R_EXPEC_LCB_SUMOUTLIER][i,j] = rfunc(datotal[MEAN][i,j].item() - datotal[MIN_OUTLIER][i,j].item()) + rfunc(datotal[MEAN][i,j].item() + datotal[MAX_OUTLIER][i,j].item()) - np.sqrt(rdatotal.Var[0,0]).item()
                # r[P_OUTLIER][i,j] = min(rfunc(datotal[MEAN][i,j] - datotal[MIN_OUTLIER][i,j]), rfunc(datotal[MEAN][i,j] + datotal[MAX_OUTLIER][i,j]))
                # r[P_OUTLIER_DTHETAWISE][i,j] = min(rfunc(datotal[MEAN][i,j] - datotal[MIN_OUTLIER_DTHETAWISE][i,j]), rfunc(datotal[MEAN][i,j] + datotal[MAX_OUTLIER_DTHETAWISE][i,j]))
                r[P_MIN][i,j] = rfunc(datotal[MIN][i,j])
                r[P_MAX][i,j] = rfunc(datotal[MAX][i,j])
                r[P_MINMAX][i,j] = min(r[P_MIN][i,j], r[P_MAX][i,j])
                r[P_MIN_DTHETAWISE][i,j] = rfunc(datotal[MIN_DTHETAWISE][i,j])
                r[P_MAX_DTHETAWISE][i,j] = rfunc(datotal[MAX_DTHETAWISE][i,j])
                r[P_MINMAX_DTHETAWISE][i,j] = min(r[P_MIN_DTHETAWISE][i,j], r[P_MAX_DTHETAWISE][i,j])
                r[P_MINMAX_DIFF][i,j] = abs(r[P_MIN_DTHETAWISE][i,j] - r[P_MAX_DTHETAWISE][i,j])
                r["95"][i,j] = rfunc(datotal["95"][i,j])
                r["05"][i,j] = rfunc(datotal["05"][i,j])
                r["90"][i,j] = rfunc(datotal["90"][i,j])
                r["10"][i,j] = rfunc(datotal["10"][i,j])
                # r[P_0595_DIFF][i,j] = abs(r["95"][i,j] - r["05"][i,j])
                r[P_1090_SUM][i,j] = -(r["90"][i,j] + r["10"][i,j])
                r[P_0595_SUM][i,j] = -(r["95"][i,j] + r["05"][i,j])
                r[P_MINMAX_SUM][i,j] = -(r[P_MIN][i,j] + r[P_MAX][i,j])
        
        check_or_create_dir(src_path)
        for k,v in datotal.items():
            np.save(src_path+"{}.npy".format(k), v)
        for r_type in r_types:
            np.save(src_path+"{}.npy".format(r_type), r[r_type])
        
        # np.save(SRC_PATH+"datotal.npy", datotal[TRUE])
        # np.save(SRC_PATH+"datotal_mean.npy", datotal[MEAN])
        # np.save(SRC_PATH+"datotal_sigma.npy", datotal[SIGMA])
        # np.save(SRC_PATH+"datotal_median.npy", datotal[MEDIAN])
        # np.save(SRC_PATH+"datotal_min_outlier.npy", datotal[MIN_OUTLIER])
        # np.save(SRC_PATH+"datotal_max_outlier.npy", datotal[MAX_OUTLIER])
        # np.save(SRC_PATH+"r_true.npy", r[R_TRUE])
        # np.save(SRC_PATH+"r_only_mean.npy", r[ONLY_MEAN])
        # np.save(SRC_PATH+"r_expec.npy", r[R_EXPEC])
        # np.save(SRC_PATH+"r_expec_lcb.npy", r[R_EXPEC_LCB])
        # np.save(SRC_PATH+"r_expec_min_outlier.npy", r[R_EXPEC_MINOUTLIER])
        # np.save(SRC_PATH+"r_expec_max_outlier.npy", r[R_EXPEC_MAXOUTLIER])
        # np.save(SRC_PATH+"r_expec_lcb_min_outlier.npy", r[R_EXPEC_LCB_MINOUTLIER])
        # np.save(SRC_PATH+"r_expec_outlier.npy", r[R_EXPEC_OUTLIER])
        # np.save(SRC_PATH+"r_expec_sum_outlier.npy", r[R_EXPEC_SUMOUTLIER])
        # np.save(SRC_PATH+"r_expec_lcb_outlier.npy", r[R_EXPEC_LCB_OUTLIER])
        # np.save(SRC_PATH+"r_expec_lcb_sum_outlier.npy", r[R_EXPEC_LCB_SUMOUTLIER])
    else:
        # datotal[TRUE] = np.load(src_path+"datotal.npy")
        # datotal[MEAN] = np.load(src_path+"datotal_mean.npy")
        # datotal[SIGMA] = np.load(src_path+"datotal_sigma.npy")
        # datotal[MEDIAN] = np.load(src_path+"datotal_median.npy")
        # datotal[MIN_OUTLIER] = np.load(SRC_PATH+"datotal_min_outlier.npy")
        # datotal[MAX_OUTLIER] = np.load(SRC_PATH+"datotal_max_outlier.npy")
        # datotal[MIN_OUTLIER_DTHETAWISE] = np.load(src_path+"min_outlier_dtehatwise.npy")
        # datotal[MAX_OUTLIER_DTHETAWISE] = np.load(src_path+"max_outlier_dtehatwise.npy")
        
        r_types = [
            R_TRUE, 
            #    ONLY_MEAN, 
               R_EXPEC, 
            #    R_STD,
            #    "90",
            #    "10",
            #    P_MIN,
            #    P_MAX,
            #    P_MINMAX,
            #    P_MINMAX_DIFF,
                P_MINMAX_SUM,
            #    P_1090_SUM,
               P_0595_SUM,
            #    P_MIN_DTHETAWISE,
            #    P_MAX_DTHETAWISE,
            #    P_MINMAX_DTHETAWISE,
        ]
        r[P_MINMAX_SUM+"_agerage10"] = cv2.filter2D(r[P_MINMAX_SUM], -1, np.full((5,5),1./(5**2)))
        for r_type in r_types:
            r[r_type] = np.load(src_path+"/{}.npy".format(r_type))
        for gain in [1.0, 2.0, 4.0, 8.0, 20.0]:
        #     # name = P_0595_SUM+str(gain)
        #     # r[name] = r[R_EXPEC] - gain*r[P_0595_SUM]
        #     # r_types.append(name)
        #     # name = P_MINMAX_SUM+str(gain)
        #     # r[name] = r[R_EXPEC] - gain*r[P_MINMAX_SUM]
        #     # r_types.append(name)
        #     name = P_MINMAX_SUM+"_3average"+str(gain)
        #     r[name] = r[R_EXPEC] - gain * cv2.filter2D(r[P_MINMAX_SUM], -1, np.full((3,3),1./(3**2)))
        #     r_types.append(name)
        #     name = P_MINMAX_SUM+"_5average"+str(gain)
        #     r[name] = r[R_EXPEC] - gain * cv2.filter2D(r[P_MINMAX_SUM], -1, np.full((5,5),1./(5**2)))
        #     r_types.append(name)
            name = P_MINMAX_SUM+"average_mms"+str(gain)
            tmp = cv2.filter2D(r[P_MINMAX_SUM], -1, np.full((filter_kernel_size,filter_kernel_size),1./(filter_kernel_size**2)))
            r[name] = (tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))
            r[name] = r[R_EXPEC] - gain * r[name]
            # r[name] = r[R_EXPEC] * (1 + gain * r[name])
            r_types.append(name)
        name = P_MINMAX_SUM+"average_mms"
        tmp = cv2.filter2D(r[P_MINMAX_SUM], -1, np.full((filter_kernel_size,filter_kernel_size),1./(filter_kernel_size**2)))
        r[name] = - (tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))*2
        r_types.append(name)
    
            
    #     r[R_TRUE] = np.load(SRC_PATH+"r_true.npy")
    #     r[ONLY_MEAN] = np.load(SRC_PATH+"r_only_mean.npy")
    #     r[R_EXPEC] = np.load(SRC_PATH+"r_expec.npy")
    #     r[R_EXPEC_LCB] = np.load(SRC_PATH+"r_expec_lcb.npy")
    #     r[R_EXPEC_MINOUTLIER] = np.load(SRC_PATH+"r_expec_min_outlier.npy")
    #     r[R_EXPEC_MAXOUTLIER] = np.load(SRC_PATH+"r_expec_max_outlier.npy")
    #     r[R_EXPEC_LCB_MINOUTLIER] = np.load(SRC_PATH+"r_expec_lcb_min_outlier.npy")
    #     r[R_EXPEC_OUTLIER] = np.load(SRC_PATH+"r_expec_outlier.npy")
    #     r[R_EXPEC_SUMOUTLIER] = np.load(SRC_PATH+"r_expec_sum_outlier.npy")
    #     r[R_EXPEC_LCB_OUTLIER] = np.load(SRC_PATH+"r_expec_lcb_outlier.npy")
    #     r[R_EXPEC_LCB_SUMOUTLIER] = np.load(SRC_PATH+"r_expec_lcb_sum_outlier.npy")

    rmatrix = lambda r_type1, r_type2: [r[r_type2][i_policy,i_smsz] for i_policy,i_smsz in zip(np.argmax(r[r_type1], axis=0), range(len(r[R_TRUE])))]
    for r_type in r_types:
        policy[r_type] = pd.DataFrame({
            "smsz": smsz,
            "best_dtheta2": np.array(dtheta2)[np.argmax(r[r_type], axis=0)],
            "est_return": rmatrix(r_type, r_type),
            "true_return": rmatrix(r_type, R_TRUE)
        })
        
    colors = ["blue", "orange", "red", "green", "purple", "purple", "purple", "purple", "purple", "purple", "purple", "orange", "red", "green", "purple"]
    addpos = {0: (0,-0.0025), 1: (0.0001,-0.0025), 2: (0, 0.0025), 3: (0.0001, 0.0025), 4:(0.0002, -0.0025), 5:(0.0002, 0.0025), 6:(-0.0001,-0.0025), 7:(-0.0001,0.0025), 8:(-0.0002,-0.0025), 9:(-0.0003,0.0025), 10:(-0.0003,-0.0025), 11:(-0.0004,0.0025), 12:(-0.0004,-0.0025)}
    visc_title_pair = [
        ([True]*len(r_types), "all"),
        ([False]*len(r_types), "none"),
        # ([False,False,True,True,False,False,False,False,False,False,False], "r_expec & r_expec_lcb"),
        # ([False,False,False,True,True,False,False,False,False,False,False], "r_expec_lcb & r_expec_min_outlier"),
        # ([False,False,False,False,True,True,False,False,False,False,False], "r_expec_min_outlier & r_expec_lcb_min_outlier"),
        # ([False,False,False,False,True,False,True,False,False,False,False], "r_expec_min_outlier & r_expec_outlier"),
        # ([False,False,False,False,True,False,False,True,False,False,False], "r_expec_min_outlier & r_expec_max_outlier"),
        # ([False,False,False,False,True,False,True,True,False,False,False], "r_expec_ both & min & max_outlier"),
        # ([False,False,False,True,False,False,True,False,False,False,False], "r_expec_lcb & _outlier"),
        # ([False,False,True,True,False,False,True,False,False,False,False], "r_expec & _lcb & _outlier"),
    ] + [([False]*i+[True]+[False]*(len(r_types)-i-1), r_type) for i,r_type in enumerate(r_types)]
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
    # for i, r_type in enumerate(r_types):
    #     fig.add_trace(
    #         go.Scatter(
    #             x=[x+addpos[i][0] for x in policy[r_type]["smsz"]], y=[y+addpos[i][1] for y in policy[r_type]["best_dtheta2"]],
    #             mode='markers', 
    #             name=r_type,
    #             marker_color=colors[i],
    #             # showlegend=False,
    #         )
    #     )
    # fig['layout']['updatemenus'] = [{
    #         "type": "dropdown",
    #         "buttons": [{
    #             'label': vis_title,
    #             'method': "update",
    #             'args':[
    #                 {'visible': [True]+visc},
    #             ]
    #         } for i, (visc, vis_title) in enumerate(visc_title_pair)],
    #         "active": 0,
    #     }]
    # fig['layout']['xaxis']['title'] = "size_srcmouth"
    # fig['layout']['yaxis']['title'] = "dtheta2"
    # check_or_create_dir(save_dir)
    # plotly.offline.plot(fig, filename = save_dir + "datotal_org.html", auto_open=False)
        
    fig = go.Figure()
    for i, r_type in enumerate(r_types):
        fig.add_trace(
            go.Scatter(
                x=policy[r_type]["smsz"], y=[y-0.005*i for y in policy[r_type]["true_return"]],
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
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "return"
    check_or_create_dir(save_dir)
    plotly.offline.plot(fig, filename = save_dir + "return.html", auto_open=False)
    
    name = P_MINMAX_SUM+"average_mms"
    fig = go.Figure()
    fig.update_layout(hoverdistance = 5)
    fig.add_trace(go.Heatmap(
        z = r[name],
        x = smsz, y = dtheta2,
        # colorscale = "Greys",
        # zmin = -1.0, zmax = 0,
        # reversescale = True,
        colorbar=dict(
            title=name,
            titleside="top", ticks="outside",
            x = 1.1,
        ),
    ))
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
    fig['layout']['updatemenus'] = [{
            "type": "dropdown",
            "buttons": [{
                'label': vis_title,
                'method': "update",
                'args':[
                    {'visible': [True]+visc},
                ]
            } for i, (visc, vis_title) in enumerate(visc_title_pair)],
            "active": 0,
        }]
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "dtheta2"
    check_or_create_dir(save_dir)
    plotly.offline.plot(fig, filename = save_dir + "{}.html".format(name), auto_open=False)
    
    # fig = go.Figure()
    # fig.update_layout(hoverdistance = 5)
    # fig.add_trace(go.Heatmap(
    #     z = datotal[MAX_OUTLIER_DTHETAWISE],
    #     x = smsz, y = dtheta2,
    #     colorscale = "Greys",
    #     # zmin = 0., zmax = 0.55,
    #     colorbar=dict(
    #         title="datotal (10 最大値 - 中央値)",
    #         titleside="top", ticks="outside",
    #         x = 1.1,
    #     ),
    # ))
    # for i, r_type in enumerate(r_types):
    #     fig.add_trace(
    #         go.Scatter(
    #             x=[x+addpos[i][0] for x in policy[r_type]["smsz"]], y=[y+addpos[i][1] for y in policy[r_type]["best_dtheta2"]],
    #             mode='markers', 
    #             name=r_type,
    #             marker_color=colors[i],
    #             # showlegend=False,
    #         )
    #     )
    # fig['layout']['updatemenus'] = [{
    #         "type": "dropdown",
    #         "buttons": [{
    #             'label': vis_title,
    #             'method': "update",
    #             'args':[
    #                 {'visible': [True]+visc},
    #             ]
    #         } for i, (visc, vis_title) in enumerate(visc_title_pair)],
    #         "active": 0,
    #     }]
    # fig['layout']['xaxis']['title'] = "size_srcmouth"
    # fig['layout']['yaxis']['title'] = "dtheta2"
    # plotly.offline.plot(fig, filename = save_dir + "datotal_max_outlier_dthetawise.html", auto_open=False)
    
    # for r_type_map in r_types:
    #     fig = go.Figure()
    #     fig.update_layout(hoverdistance = 5)
    #     fig.add_trace(go.Heatmap(
    #         z = r[r_type_map],
    #         x = smsz, y = dtheta2,
    #         colorscale = "Greys",
    #         reversescale = True,
    #         zmin = -1.0, zmax = 0,
    #         colorbar=dict(
    #             title=r_type_map,
    #             titleside="top", ticks="outside",
    #             x = 1.1,
    #         ),
    #     ))
    #     for i, r_type in enumerate(r_types):
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=[x+addpos[i][0] for x in policy[r_type]["smsz"]], y=[y+addpos[i][1] for y in policy[r_type]["best_dtheta2"]],
    #                 mode='markers', 
    #                 name=r_type,
    #                 marker_color=colors[i],
    #                 # showlegend=False,
    #             )
    #         )
    #     fig['layout']['updatemenus'] = [{
    #             "type": "dropdown",
    #             "buttons": [{
    #                 'label': vis_title,
    #                 'method': "update",
    #                 'args':[
    #                     {'visible': [True]+visc},
    #                 ]
    #             } for i, (visc, vis_title) in enumerate(visc_title_pair)],
    #             "active": 0,
    #         }]
    #     fig['layout']['xaxis']['title'] = "size_srcmouth"
    #     fig['layout']['yaxis']['title'] = "dtheta2"
    #     check_or_create_dir(save_dir)
    #     plotly.offline.plot(fig, filename = save_dir + r_type_map + ".html", auto_open=False)