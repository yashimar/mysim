from core_tool import *
from .greedyopt import *
from copy import deepcopy


def Help():
    pass


def Run(ct, *args):
    r_types = (P_MINMAX_SUM, R_TRUE, R_EXPEC, R_STD,)
    src_path = SRC_PATH + "3_3"
    save_dir = PICTURE_DIR + "curriculum5/c1/trues_sampling/tip_ketchup_smsz_dthtea2".replace("/","_") + "/greedy_opt/{}_{}/optcheck/".format(3, 3)
    
    smsz = np.linspace(0.03,0.08,100)
    dtheta2 = np.linspace(0.1,1,100)[::-1]
    r = dict()
    for r_type in r_types:
        r[r_type] = np.load(src_path+"/{}.npy".format(r_type))
    # r[P_MINMAX_SUM+"05"] = r[R_EXPEC] - 0.5*r[P_MINMAX_SUM]
    # PA = P_MINMAX_SUM+"_5average"
    # r[PA] = cv2.filter2D(r[P_MINMAX_SUM], -1, np.full((5,5),1./(5**2)))
    # r[PA+"005"] = r[R_EXPEC] - 0.05*r[PA]
    name = P_MINMAX_SUM+"_average_mms"
    tmp = cv2.filter2D(r[P_MINMAX_SUM], -1, np.full((3,3),1./(3**2)))
    r[name] = (tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))
    r[name+"2.0"] = r[R_EXPEC] - 2.0 * r[name]
        
    def format_output(r_type):
        r_col = r[r_type][:,smsz_idx]
        opt_dtheta_idx = np.argmax(r_col)
        opt_dtheta = dtheta2[opt_dtheta_idx]
        max_r_col = max(r_col)
        max80_r_col = sorted(r_col)[-20]
        return r_col, opt_dtheta, max_r_col, max80_r_col, opt_dtheta_idx

    smsz_idx_list = range(0,100)
    vis_min = -8
    trace = defaultdict(list)
    for smsz_idx in smsz_idx_list:
        r_col, opt_dtheta, max_r_col, max80_r_col, _ = format_output(R_TRUE)
        trace[0].append(go.Scatter(
            x=dtheta2, y=r_col,
            mode='markers', 
            name=R_TRUE,
            visible=False
        ))
        trace[1].append(go.Scatter(x=[min(dtheta2), max(dtheta2)], y=[max80_r_col,max80_r_col], name="80 percentile [R_TRUE]", line=dict(color="gray", dash="dash"),visible=False))
        trace[2].append(go.Scatter(x=[opt_dtheta, opt_dtheta], y=[vis_min,max_r_col], line=dict(color="blue", dash="dot"),visible=False,showlegend=False))
        trace[3].append(go.Scatter(
            x=[opt_dtheta], y=[max_r_col],
            mode='markers', 
            name="best reward [{}]".format(R_TRUE),
            marker_color="red",
            marker_size=10,
            showlegend=False,
            visible=False
        ))
        
        # r_type_save = P_MINMAX_SUM+"_average_mms"+"2.0"
        # r_type = P_MINMAX_SUM+"_average_mms"+"2.0"
        r_type_save = R_EXPEC
        r_type = R_EXPEC
        color = "purple"
        r_col, opt_dtheta, max_r_col, max80_r_col, opt_dtheta_idx = format_output(r_type)
        trace[4].append(go.Scatter(
            x=dtheta2, y=r_col,
            mode='lines', 
            name=r_type,
            line=dict(color=color, dash="dash"),
            # marker_color=color,
            # marker_size=5,
            visible=False,
            # error_y=dict(
            #     type="data",
            #     symmetric=True,
            #     array=r[R_STD][:,smsz_idx],
            #     # array=[2]*100,
            #     color=color,
            #     thickness=1.5,
            #     width=3,
            # ),
        ))
        trace[5].append(go.Scatter(x=[opt_dtheta, opt_dtheta], y=[vis_min,max_r_col], line=dict(color=color, dash="dot"),visible=False,showlegend=False))
        trace[6].append(go.Scatter(
            x=[opt_dtheta], y=[max_r_col],
            mode='markers', 
            name="best reward [{}]".format(r_type),
            marker_color="red",
            marker_size=10,
            showlegend=False,
            visible=False
        ))
        trace[7].append(go.Scatter(
            x=[opt_dtheta], y=[r[R_TRUE][:,smsz_idx][opt_dtheta_idx]],
            mode='markers', 
            name="true reward at opt_dtheta [{}]".format(r_type),
            marker_color="purple",
            marker_size=10,
            showlegend=False,
            visible=False,
        ))
        
        r_type = R_EXPEC
        color = "orange"
        r_col, opt_dtheta, max_r_col, max80_r_col, opt_dtheta_idx = format_output(r_type)
        trace[8].append(go.Scatter(
            x=dtheta2, y=r_col,
            mode='markers', 
            name=r_type,
            marker_color=color,
            marker_size=5,
            visible=False,
            error_y=dict(
                type="data",
                symmetric=True,
                array=r[R_STD][:,smsz_idx],
                # array=[2]*100,
                color=color,
                thickness=1.5,
                width=3,
            ),
        ))
        trace[9].append(go.Scatter(x=[opt_dtheta, opt_dtheta], y=[vis_min,max_r_col], line=dict(color=color, dash="dot"),visible=False,showlegend=False))
        trace[10].append(go.Scatter(
            x=[opt_dtheta], y=[max_r_col],
            mode='markers', 
            name="best reward [{}]".format(r_type),
            marker_color="red",
            marker_size=10,
            showlegend=False,
            visible=False
        ))
        trace[11].append(go.Scatter(
            x=[opt_dtheta], y=[r[R_TRUE][:,smsz_idx][opt_dtheta_idx]],
            mode='markers', 
            name="true reward at opt_dtheta [{}]".format(r_type),
            marker_color="orange",
            marker_size=10,
            showlegend=False,
            visible=False,
        ))
        
    
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    
    steps = []
    for i in range(len(smsz_idx_list)):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(smsz_idx_list)
            trace["vis{}".format(j)][i] = True
        step = dict(
            method="update",
            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                {"title": "size_srcmouth: {:.4f}".format(smsz[i])}],
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
    fig['layout']['yaxis']['title'] = "reward"
    fig['layout']['yaxis']['range'] = (vis_min,0.5)
    for i in smsz_idx_list:
        fig['layout']['sliders'][0]['steps'][i]['label'] = round(smsz[i],4)
    
    check_or_create_dir(save_dir)
    plotly.offline.plot(fig, filename = save_dir + "{}.html".format(r_type_save), auto_open=False)
    