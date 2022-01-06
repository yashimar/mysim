#coding: UTF-8
from yaml import serialize
from learn3 import *
import cv2
from scipy import interpolate
from sklearn.neural_network import MLPClassifier
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
from scipy.stats import multivariate_normal


def load_data(name, ver = 2):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    # name = "GMMSig5LCB3/t1"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/".format(ver, name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
    
    return dm, log, datotal, gmmpred, evaluation


def detect_edge(name, ver = 2):
    dm, log, datotal, gmmpred, evaluation = load_data(name, ver)
    
    eval_tip = evaluation[TIP]
    eval_tip = np.maximum(eval_tip, -3)
    
    kernel_x = np.array([
        [0,-1,0],
        [0,0,0],
        [0,1,0]
    ])
    kernel_y = np.array([
        [0,0,0],
        [-1,0,1],
        [0,0,0]
    ])
    eval_tip_edge = np.sqrt(cv2.filter2D(eval_tip, -1, kernel_x)**2 + cv2.filter2D(eval_tip, -1, kernel_y)**2)
    eval_tip_edge = (eval_tip_edge - eval_tip_edge.min()) / (eval_tip_edge.max() - eval_tip_edge.min())
    
    y = np.max(eval_tip_edge, axis=0)
    fitted_curve = interpolate.interp1d(dm.smsz, y, kind='cubic')
    
    x = np.linspace(0.3,0.8,1000)
    fy = np.maximum(fitted_curve(x), 0)
    
    fig = plt.figure()
    plt.scatter(dm.smsz, fitted_curve(dm.smsz))
    plt.plot(dm.smsz, y)
    plt.show()
    
    fig = plt.figure()
    plt.hist(np.random.choice(x, size=(10000), p=fy/np.sum(fy)), bins=40)
    plt.show()
        
    # fig = go.Figure()
    # fig.add_trace(go.Heatmap(
    #             z = eval_tip_edge, x = dm.smsz, y = dm.dtheta2,
    #             # colorscale = cs if cs != None else "Viridis",
    #             # zmin = zmin, zmax = zmax,
    #             # colorbar=dict(
    #             #     titleside="top", ticks="outside",
    #             #     x = posx_set[col-1], y = posy_set[row-1],
    #             #     thickness=23, len = clength,
    #             #     # tickcolor = "black",
    #             #     tickfont = dict(color = "black"),
    #             # ),
    # ))
    # fig.show()


def detect_edge2(dm, eval_tip):
    r_thr = -1
    binary_thr = 0.
    search_size = 2
    kernel_size = 6
    sd_gain = 1
    
    X_train = np.array([[dtheta,smsz] for smsz in dm.smsz for dtheta in dm.dtheta2])
    y_train = np.where(eval_tip>r_thr,1,0).reshape(-1,)

    clf = MLPClassifier(
        random_state=1, max_iter=1000, hidden_layer_sizes=(200,200,200,200,), tol=1e-4,
        learning_rate='adaptive', early_stopping=False, nesterovs_momentum=False,
        shuffle=True)
    clf = clf.fit(X_train, y_train)
    # pred =  clf.predict(X_train).reshape(100,100)
    pred = clf.predict_proba(X_train)
    pred = pred[:,1].reshape(100,100)
    pred_edge = np.where((0.25<pred)&(pred<0.75),1,0)
    
    Z = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            ref_array = eval_tip[max(i-search_size,0):i+search_size,max(j-search_size,0):j+search_size]
            # ref_array = pred[max(i-search_size,0):i+search_size,max(j-search_size,0):j+search_size]
            
            x,y = np.mgrid[-1:1:2./kernel_size,-1:1:2./kernel_size]
            pos = np.dstack((x,y))
            # sd_x = np.mean(np.std(ref_array, axis = 0))*sd_gain
            # sd_y = np.mean(np.std(ref_array, axis = 1))*sd_gain
            sd_x = abs(np.mean(ref_array[:,-1] - np.mean(ref_array[:,0])))*sd_gain
            sd_y = abs(np.mean(ref_array[-1,:] - np.mean(ref_array[0,:])))*sd_gain
            S = [[sd_x,0],
                 [0,sd_y]]
            rv = multivariate_normal(mean=[0.,0.], cov=S)
            kernel = rv.pdf(pos)
            
            
            # # kernel_size = 2
            # kernel_size = int(np.std(tmp_eval_array)*1)
            # # print(np.std(tmp_eval_array))
            # kernel_size = max(kernel_size, 2)
            # if kernel_size % 2 == 1:
            #     kernel_size += 1
            
            tmp_w_min = max(j-kernel_size/2,0)
            tmp_w_max = min(j+kernel_size/2,100)
            tmp_h_min = max(i-kernel_size/2,0)
            tmp_h_max = min(i+kernel_size/2,100)
            
            pad_h_top = max(kernel_size/2-i,0)
            pad_h_bottom = max(kernel_size/2+i-100,0)
            pad_w_left = max(kernel_size/2-j,0)
            pad_w_right = max(kernel_size/2+j-100,0)
            
            tmp_array = pred_edge[tmp_h_min:tmp_h_max,tmp_w_min:tmp_w_max]
            # print(tmp_array)
            # print(tmp_array.shape, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
            
            
            tmp_array = np.pad(tmp_array, ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), 'constant')
            
            
            
            # kernel = np.ones((kernel_size,kernel_size))/((tmp_w_max - tmp_w_min)*(tmp_h_max - tmp_h_min))
            # kernel = np.ones((tmp_w_max - tmp_w_min,tmp_h_max - tmp_h_min))/((tmp_w_max - tmp_w_min)*(tmp_h_max - tmp_h_min))
            # print(i,j,tmp_array.shape,kernel.shape)
            Z[i,j] = np.sum(np.multiply(tmp_array, kernel))
    pred_edge2 = Z.copy()
    pred_edge2_binary = np.where(pred_edge2>binary_thr,1,0)
    fixed_eval = eval_tip - pred_edge2_binary*100
    fixed_eval = np.maximum(fixed_eval, -5)
    
    return pred, pred_edge, pred_edge2, pred_edge2_binary, fixed_eval, X_train, y_train


def detect_edge3(dm, eval_tip):
    r_thr = -1.0
    eval_thr = -5
    binary_thr = 0.
    pred_thr = 30 #%
    search_size = 15
    kernel_size = 100
    var_gain = 1.5*1e-4
    # sd_gain = 1
    kernel_thr = 0.01
    
    X_train = np.array([[dtheta,smsz] for smsz in dm.smsz for dtheta in dm.dtheta2])
    y_train = np.where(eval_tip>r_thr,1,0).reshape(-1,)

    clf = MLPClassifier(
        random_state=1, max_iter=1000, hidden_layer_sizes=(200,200,200,200,200,200,200,), tol=1e-4,
        learning_rate='adaptive', early_stopping=False, nesterovs_momentum=False,
        shuffle=True)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_train)
    pred = pred[:,1].reshape(100,100)
    pred_edge = np.where((0.5-pred_thr/100./2<pred)&(pred<0.5+pred_thr/100./2),1,0)
    
    Z2 = np.zeros((100,100))
    Z3 = np.zeros((100,100))
    Z = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            if pred_edge[i,j] == 0:
                continue
            ref_array = eval_tip[max(i-search_size,0):i+search_size,max(j-search_size,0):j+search_size]
            # ref_array = pred[max(i-search_size,0):i+search_size,max(j-search_size,0):j+search_size]
            
            x,y = np.mgrid[-1:1:2./kernel_size,-1:1:2./kernel_size]
            pos = np.dstack((x,y))
            # sd_x = np.mean(np.std(ref_array, axis = 0))*sd_gain
            # sd_y = np.mean(np.std(ref_array, axis = 1))*sd_gain
            # sd_x = abs(np.mean(ref_array[:,-1] - np.mean(ref_array[:,0])))*sd_gain
            # sd_y = abs(np.mean(ref_array[-1,:] - np.mean(ref_array[0,:])))*sd_gain
            ref_array = np.maximum(ref_array, eval_thr)
            var_x = max(np.max(ref_array, axis = 1) - np.min(ref_array, axis = 1))**2*var_gain
            var_y = max(np.max(ref_array, axis = 0) - np.min(ref_array, axis = 0))**2*var_gain
            # var = (np.max(ref_array) - np.min(ref_array))**2*var_gain
            # var_x, var_y = var, var
            S = [[var_x,0],
                 [0,var_y]]
            rv = multivariate_normal(mean=[0.,0.], cov=S)
            kernel = rv.pdf(pos)
            kernel = np.where(kernel<kernel_thr,0,kernel)
            
            tmp_w_min = max(j-kernel_size/2,0)
            tmp_w_max = min(j+kernel_size/2,100)
            tmp_h_min = max(i-kernel_size/2,0)
            tmp_h_max = min(i+kernel_size/2,100)
            
            pad_h_top = max(kernel_size/2-i,0)
            pad_h_bottom = max(kernel_size/2+i-100,0)
            pad_w_left = max(kernel_size/2-j,0)
            pad_w_right = max(kernel_size/2+j-100,0)
            
            tmp_array = pred_edge[tmp_h_min:tmp_h_max,tmp_w_min:tmp_w_max]
            tmp_array = np.pad(tmp_array, ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), 'constant')
            
            # Z[i,j] = np.sum(np.multiply(tmp_array, kernel))
            woffset_min = max(kernel_size/2-j, 0)
            woffset_max = min(100+kernel_size/2-j, kernel_size)
            hoffset_min = max(kernel_size/2-i, 0)
            hoffset_max = min(100+kernel_size/2-i, kernel_size)
            # print(kernel.shape)
            # print(woffset_min,woffset_max,hoffset_min,hoffset_max)
            # print(Z[tmp_h_min:tmp_h_max,tmp_w_min:tmp_w_max].shape, kernel[hoffset_min:hoffset_max,woffset_min:woffset_max].shape)
            Z[tmp_h_min:tmp_h_max,tmp_w_min:tmp_w_max] = np.maximum(Z[tmp_h_min:tmp_h_max,tmp_w_min:tmp_w_max], kernel[hoffset_min:hoffset_max,woffset_min:woffset_max])
            Z2[i,j] = var_x
            Z3[i,j] = var_y
    pred_edge2 = Z.copy()
    pred_edge2_binary = np.where(pred_edge2>binary_thr,1,0)
    fixed_eval = eval_tip - pred_edge2_binary*100
    fixed_eval = np.maximum(fixed_eval, eval_thr)
    
    
    # fig = go.Figure()
    # fig.update_layout(
    #     height = 800,
    #     width = 800,
    # )
    # fig = make_subplots(
    #     rows=1, cols=2, 
    #     horizontal_spacing = 0.05,
    #     vertical_spacing = 0.05,
    # )
    # fig.update_layout(
    #     height = 800,
    #     width = 1800,
    # )
    # fig.add_trace(go.Heatmap(
    #     z = np.minimum(Z2,5), x = dm.smsz, y = dm.dtheta2,
    # ),1,1)
    # fig.add_trace(go.Heatmap(
    #     z = np.minimum(Z3,5), x = dm.smsz, y = dm.dtheta2,
    # ),1,2)
    # fig.show()
    
    return pred, pred_edge, pred_edge2, pred_edge2_binary, fixed_eval, X_train, y_train

    
def edge_check(name, ver = 4, edge_ver = 3):
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/".format(ver, name)
    dm, log, datotal, gmmpred, evaluation = load_data(name, ver)
    eval_tip = evaluation[TIP]
    
    if edge_ver == 2:
        pred, pred_edge, pred_edge2, pred_edge2_binary, fixed_eval, X_train, y_train = detect_edge2(dm, eval_tip)
    elif edge_ver == 3:
        pred, pred_edge, pred_edge2, pred_edge2_binary, fixed_eval, X_train, y_train = detect_edge3(dm, eval_tip)
    
    evaluation[TIP] = fixed_eval
    true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
    true_yshake = dm.datotal[SHAKE][RFUNC]
    est_ytip = np.max(evaluation[TIP], axis=0)
    est_yshake = evaluation[SHAKE]
    
    
    est_ytip_normal = np.max(eval_tip, axis=0)
    est_ytip_edge = np.max(fixed_eval, axis=0)
    opt_smsz_normal = []
    opt_dtheta_normal = []
    opt_smsz_edge = []
    opt_dtheta_edge = []
    for i in range(100):
        if est_ytip_normal[i] >= est_yshake[i]:
            opt_smsz_normal.append(dm.smsz[i])
            opt_dtheta_normal.append(dm.dtheta2[np.argmax(eval_tip[:,i])])
        if est_ytip_edge[i] >= est_yshake[i]:
            opt_smsz_edge.append(dm.smsz[i])
            opt_dtheta_edge.append(dm.dtheta2[np.argmax(fixed_eval[:,i])])
            
    tip_idx = [i for i,skill in enumerate(log['skill']) if skill == 'tip']
    scatter_dtheta2 = np.array(log['est_optparam'])[tip_idx]
    scatter_smsz = np.array(log['smsz'])[tip_idx]
            
    
    evalcs = [
        [0, "rgb(120, 120, 255)"],
        [0.8, "rgb(150, 150, 0)"],
        [1, "rgb(0, 255, 0)"],
    ]
    
    fig = make_subplots(
        rows=5, cols=2, 
        subplot_titles=[
            "評価関数", "二値化 (thr=-1)", 
            "分類NN予測",'分類NN予測 エッジ',
            'エッジ ぼかし', 'エッジ 二値化',
            '修正語評価関数','評価関数',
            '修正語評価関数','評価関数',
        ],
        horizontal_spacing = 0.05,
        vertical_spacing = 0.05,
    )
    fig.update_layout(
        height = 3000,
        width = 1800,
    )
    fig.add_trace(go.Heatmap(
        z = np.maximum(eval_tip,-5), x = dm.smsz, y = dm.dtheta2,
        colorscale = evalcs,
        colorbar=dict(
                    titleside="top", ticks="outside",
                    x = 0.475, y = 0.1,
                    thickness=23, len = 0.15,
                ),
    ),1,1)
    fig.add_trace(go.Heatmap(
        z = y_train.reshape(100,100), x = dm.smsz, y = dm.dtheta2,
        colorbar=dict(
                    titleside="top", ticks="outside",
                    x = 1, y = 0.1,
                    thickness=23, len = 0.15,
                ),
    ),1,2)
    fig.add_trace(go.Heatmap(
        z = pred, x = dm.smsz, y = dm.dtheta2,
        colorbar=dict(
                    titleside="top", ticks="outside",
                    x = 0.475, y = 0.25,
                    thickness=23, len = 0.15,
                ),
    ),2,1)
    fig.add_trace(go.Heatmap(
        z = pred_edge, x = dm.smsz, y = dm.dtheta2,
        colorbar=dict(
                    titleside="top", ticks="outside",
                    x = 1, y = 0.25,
                    thickness=23, len = 0.15,
                ),
    ),2,2)
    fig.add_trace(go.Heatmap(
        z = pred_edge2, x = dm.smsz, y = dm.dtheta2,
        colorbar=dict(
                    titleside="top", ticks="outside",
                    x = 0.475, y = 0.4,
                    thickness=23, len = 0.15,
                ),
    ),3,1)
    fig.add_trace(go.Heatmap(
        z = pred_edge2_binary, x = dm.smsz, y = dm.dtheta2,
        colorbar=dict(
                    titleside="top", ticks="outside",
                    x = 1, y = 0.4,
                    thickness=23, len = 0.15,
                ),
    ),3,2)
    fig.add_trace(go.Heatmap(
        z = fixed_eval, x = dm.smsz, y = dm.dtheta2,
        colorscale = evalcs,
        colorbar=dict(
                    titleside="top", ticks="outside",
                    x = 0.475, y = 0.55,
                    thickness=23, len = 0.15,
                ),
    ),4,1)
    fig.add_trace(go.Scatter(
        x = opt_smsz_edge, y = opt_dtheta_edge,
        mode = 'markers',
        marker = dict(
          color = 'red',  
        ),
    ),4,1)
    fig.add_trace(go.Heatmap(
        z = np.maximum(eval_tip,-5), x = dm.smsz, y = dm.dtheta2,
        colorscale = evalcs,
        colorbar=dict(
                    titleside="top", ticks="outside",
                    x = 1, y = 0.55,
                    thickness=23, len = 0.15,
                ),
    ),4,2)
    fig.add_trace(go.Scatter(
        x = opt_smsz_normal, y = opt_dtheta_normal,
        mode = 'markers',
        marker = dict(
          color = 'red',  
        ),
    ),4,2)
    fig.add_trace(go.Heatmap(
        z = fixed_eval, x = dm.smsz, y = dm.dtheta2,
        colorscale = evalcs,
        colorbar=dict(
                    titleside="top", ticks="outside",
                    x = 0.475, y = 0.7,
                    thickness=23, len = 0.15,
                ),
    ),5,1)
    fig.add_trace(go.Scatter(
        x = opt_smsz_edge, y = opt_dtheta_edge,
        mode = 'markers',
        marker = dict(
          color = 'red',  
        ),
    ),5,1)
    fig.add_trace(go.Scatter(
        x = scatter_smsz, y = scatter_dtheta2,
        mode = 'markers',
        marker = dict(
          color = 'white',  
        ),
    ),5,1)
    fig.add_trace(go.Heatmap(
        z = np.maximum(eval_tip,-5), x = dm.smsz, y = dm.dtheta2,
        colorscale = evalcs,
        colorbar=dict(
                    titleside="top", ticks="outside",
                    x = 1, y = 0.7,
                    thickness=23, len = 0.15,
                ),
    ),5,2)
    fig.add_trace(go.Scatter(
        x = opt_smsz_normal, y = opt_dtheta_normal,
        mode = 'markers',
        marker = dict(
          color = 'red',  
        ),
    ),5,2)
    fig.add_trace(go.Scatter(
        x = scatter_smsz, y = scatter_dtheta2,
        mode = 'markers',
        marker = dict(
          color = 'white',  
        ),
    ),5,2)
    # fig.show()
    plotly.offline.plot(fig, filename = save_img_dir + "edge.html", auto_open=False)
    
    
    optr = [true_ytip[i] if yt > ys else true_yshake[i] for i, (yt, ys) in enumerate(zip(est_ytip, est_yshake))]
    color = ["red" if yt > ys else "purple" for yt, ys in zip(est_ytip, est_yshake)]
    
    fig = go.Figure()
    fig.update_layout(
        # margin=dict(t=20,b=10),
        width = 1600,
        height = 900,
    )
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = optr,
        mode = "markers",
        name = "reward (shake) at est optparam",
        showlegend = False,
        marker = dict(size = 16, color = color),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = est_ytip,
        # x = dm.smsz[:79].tolist()+[0.7], y = est_ytip[:79].tolist()+[-4],
        mode = "lines",
        name = "evaluatioin (tip) at est optparam",
        showlegend = False,
        line = dict(width = 4, color = "red"),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = est_yshake,
        mode = "lines",
        name = "evaluation (shake) at est optparam",
        showlegend = False,
        line = dict(width = 4, color = "purple"),
    ))
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['xaxis']['range'] = (0.29,0.82)
    fig['layout']['yaxis']['range'] = (-4,0.2)
    fig['layout']['yaxis']['dtick'] = 1
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 40
    fig['layout']['xaxis']['linewidth'] = 5
    fig['layout']['yaxis']['linewidth'] = 5
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        width = 1600,
        height = 600,
    )
    plotly.offline.plot(fig, filename = save_img_dir + "comp_use_edge.html", auto_open=False)


def opttest_comp_use_edge(name, n, ch = None, ver = 2, edge_ver = 3):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/opttest_comp/{}/".format(ver, name, ch)
    check_or_create_dir(save_img_dir)
    
    y_concat = []
    yest_concat = {TIP: [], SHAKE: []}
    for i in range(1,n):
    # for i in range(1,25)+range(26,80)+range(90,96)+range(98,100):
        logdir = basedir + "{}/t{}/{}".format(name, i, ch)
        print(logdir)
        dm = Domain3.load(logdir+"dm.pickle")
        datotal = setup_datotal(dm, logdir)
        gmmpred = setup_gmmpred(dm, logdir)
        evaluation = setup_eval(dm, logdir)
        
        if edge_ver == 2:
            pred, pred_edge, pred_edge2, pred_edge2_binary, fixed_eval, X_train, y_train = detect_edge2(dm, evaluation[TIP])
        elif edge_ver == 3:
            pred, pred_edge, pred_edge2, pred_edge2_binary, fixed_eval, X_train, y_train = detect_edge3(dm, evaluation[TIP])
        elif edge_ver == None:
            fixed_eval = evaluation[TIP]
        evaluation[TIP] = fixed_eval
        
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        est_ytip = np.max(evaluation[TIP], axis=0)
        est_yshake = evaluation[SHAKE]
        
        y = []
        yest = {TIP: [], SHAKE: []}
        for idx_smsz in range(len(dm.smsz)):
            if est_ytip[idx_smsz] > est_yshake[idx_smsz]:
                y.append(true_ytip[idx_smsz])
            else:
                y.append(true_yshake[idx_smsz])
            yest[TIP].append(est_ytip[idx_smsz])
            yest[SHAKE].append(est_yshake[idx_smsz])
        y_concat.append(y)
        for skill in [TIP, SHAKE]:
            yest_concat[skill].append(yest[skill])
    ymean = np.mean(y_concat, axis = 0)
    ysd = np.std(y_concat, axis = 0)
    yestmean = dict()
    yestsd = dict()
    yp = dict()
    yestp = defaultdict(lambda: dict())
    for skill in [TIP, SHAKE]:
        yestmean[skill] = np.mean(yest_concat[skill], axis = 0)
        yestsd[skill] = np.std(yest_concat[skill], axis = 0)
    for p in [0,2,5,10,50,90,95,98,100,25,75]:
        yp[p] = np.percentile(y_concat, p, axis = 0)
        
        # if p in [5,10,50,90,95]:
        #     Print('{}percentile:'.format(p))
        #     for i in range(5):    
        #         Print('smsz_idx {}~{}:'.format(20*i,20*(i+1)), np.mean(yp[p][20*i:20*(i+1)]))
        
        for skill in[TIP, SHAKE]:
            yestp[skill][p] = np.percentile(np.array(yest_concat[skill]), p, axis = 0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = np.max(dm.datotal[TIP][RFUNC], axis = 0),
        mode = "lines",
        name = "reward (TIP)",
        line = dict(width = 2, color = "blue"),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = dm.datotal[SHAKE][RFUNC],
        mode = "lines",
        name = "reward (SHAKE)",
        line = dict(width = 2, color = "red"),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yp[50],
        mode = "markers",
        name = "reward at opt param (5%, 50%, 95%)",
        marker = dict(color = 'black'),
        error_y=dict(
            type="data",
            symmetric=False,
            array=yp[95]-yp[50],
            arrayminus=yp[50]-yp[5],
            thickness=1.5,
            width=3,
            color = 'black',
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = ymean,
        mode = "lines",
        name = "reward at opt param (mean)",
        line = dict(width = 3, color = "black"),
    ))
    badr = [len([yi for yi in y if yi < true_yshake[idx_smsz]]) for idx_smsz, y in enumerate(np.array(y_concat).T)]
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = np.ones(len(dm.smsz))*0.1+np.array([0.1 if i%2==0 else 0 for i in range(len(dm.smsz))]),
        mode = "lines+text",
        text = ["{:.0f}".format(1.*b/len(y_concat)*100) if b != 0 else "" for b in badr],
        line = dict(width=0),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = ymean,
        mode = "markers",
        name = "reward at opt param",
        error_y=dict(
            type="data",
            symmetric=False,
            array=np.zeros(len(ysd)),
            arrayminus=ysd,
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestmean[TIP],
        mode = "markers",
        name = "evaluation (TIP) at opt param",
        error_y=dict(
            type="data",
            symmetric=True,
            array=yestsd[TIP],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestmean[SHAKE],
        mode = "markers",
        name = "evaluation (SHAKE)",
        error_y=dict(
            type="data",
            symmetric=True,
            array=yestsd[SHAKE],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[TIP][50],
        mode = "markers",
        name = "evaluation (TIP) at opt param (0%, 100%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[TIP][100]-yestp[TIP][50],
            arrayminus=yestp[TIP][50]-yestp[TIP][0],
            thickness=0.8,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[TIP][50],
        mode = "markers",
        name = "evaluation (TIP) at opt param (5%, 50%, 95%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[TIP][95]-yestp[TIP][50],
            arrayminus=yestp[TIP][50]-yestp[TIP][5],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[SHAKE][50],
        mode = "markers",
        name = "evaluation (SHAKE) at opt param (0%, 100%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[SHAKE][100]-yestp[SHAKE][50],
            arrayminus=yestp[SHAKE][50]-yestp[SHAKE][0],
            thickness=0.8,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[SHAKE][50],
        mode = "markers",
        name = "evaluation (SHAKE) at opt param (5%, 50%, 95%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[SHAKE][95]-yestp[SHAKE][50],
            arrayminus=yestp[SHAKE][50]-yestp[SHAKE][5],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig['layout']['yaxis']['range'] = (-5,0.5)
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    
    if edge_ver != None:
        plotly.offline.plot(fig, filename = save_img_dir + "opttest_comp_use_edge.html", auto_open=False)
    else:
        plotly.offline.plot(fig, filename = save_img_dir + "opttest_comp.html", auto_open=False)
        
        
    fig = go.Figure()
    fig.update_layout(
        height = 600,
    )
    # for n_i, name in enumerate(names[::-1]):
    #     y_concat = y_concat_concat[name]
    #     ymean = ymean_concat[name]
    #     ysd = ysd_concat[name]
    #     yestmean = yestmean_concat[name]
    #     yestsd = yestsd_concat[name]
    #     yp = yp_concat[name]
    #     yestp = yestp_concat[name]
    for i in range(5):
        s_idx, e_idx = 20*i, 20*(i+1)
        fig.add_trace(go.Box(
                # y0 = np.array(dm.smsz)[j] 
                # y0 = n_i,
                # y0 = vis_names[::-1][n_i],
                y0 = '{:.1f}~{:.1f}'.format(dm.smsz[s_idx], dm.smsz[e_idx-1]),
                upperfence = [np.mean(yp[95][s_idx:e_idx])],
                q3 = [np.mean(yp[90][s_idx:e_idx])],
                median = [np.mean(yp[50][s_idx:e_idx])],
                q1 = [np.mean(yp[10][s_idx:e_idx])],
                lowerfence = [np.mean(yp[5][s_idx:e_idx])],            
                # fillcolor = "white",
                marker = dict(color = 'skyblue'),
                # marker = dict(color = colors[::-1][n_i], line = dict(outliercolor = colors[::-1][n_i])),
                line = dict(width = 4),
                width = 0.4,
                # whiskerwidth = 1,
                showlegend = False,
        ))
        fig.add_trace(go.Box(
                # y0 = np.array(dm.smsz)[j] 
                # y0 = n_i,
                # y0 = vis_names[::-1][n_i],
                y0 = '{:.1f}~{:.1f}'.format(dm.smsz[s_idx], dm.smsz[e_idx-1]),
                upperfence = [np.mean(yp[95][s_idx:e_idx])],
                q3 = [np.mean(yp[75][s_idx:e_idx])],
                median = [np.mean(yp[50][s_idx:e_idx])],
                q1 = [np.mean(yp[25][s_idx:e_idx])],
                lowerfence = [np.mean(yp[5][s_idx:e_idx])],            
                fillcolor = "white",
                marker = dict(color = 'blue'),
                # marker = dict(color = colors[::-1][n_i], line = dict(outliercolor = colors[::-1][n_i])),
                line = dict(width = 12),
                width = 0.8,
                # whiskerwidth = 1,
                showlegend = False,
        ))
    fig['layout']['xaxis']['range'] = (-2.8,0)
    fig['layout']['xaxis']['title'] = "reward"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
    )
    
    if edge_ver == None:
        plotly.offline.plot(fig, filename = save_img_dir + "bar.html", auto_open=False)
    else:
        plotly.offline.plot(fig, filename = save_img_dir + "bar_use_edge.html", auto_open=False)
    
    for y in ymean:
        print(y)
    

def Run(ct,*args):
    # name = "GMM12Sig8LCB4/checkpoints/t{}/ch100".format(args[0])
    # detect_edge(name, ver = 3)
    # detect_edge2(name, ver = 3)
    # edge_check(name, ver = 3)
    
    # i = args[0]
    # name = "GMM12Sig8LCB4/checkpoints/t{}/ch100".format(i)
    # edge_check(name, ver = 4, edge_ver = 3)
    
    # name = "GMM12Sig8LCB4_def/checkpoints/t{}/u1add50".format(i)
    # edge_check(name, ver = 4, edge_ver = 3)
    
    # name = "GMM12Sig8LCB4_def/checkpoints/t{}/u2add50".format(i)
    # edge_check(name, ver = 4, edge_ver = 3)
    
    # for i in range(1,30):
    #     name = "GMM12Sig8LCB4/checkpoints/t{}/ch100".format(i)
    #     # name = "GMM12Sig8LCB4/checkpoints/t{}/u3add50".format(i)
    #     edge_check(name, ver = 4, edge_ver = 3)
        
    # for i in range(1,30):
    #     name = "GMM12Sig8LCB4_def/checkpoints/t{}/u1add50".format(i)
    #     # name = "GMM12Sig8LCB4/checkpoints/t{}/u3add50".format(i)
    #     edge_check(name, ver = 4, edge_ver = 3)
        
    # for i in range(1,30):
    #     name = "GMM12Sig8LCB4_def/checkpoints/t{}/u2add50".format(i)
    #     # name = "GMM12Sig8LCB4/checkpoints/t{}/u3add50".format(i)
    #     edge_check(name, ver = 4, edge_ver = 3)
    
    # opttest_comp_use_edge("GMM12Sig8LCB4/checkpoints", 30, ch='ch100/', ver=4)
    # opttest_comp_use_edge("GMM12Sig8LCB4/checkpoints", 30, ch='u3add50/', ver=4)
    # opttest_comp_use_edge("GMM12Sig8LCB4_def/checkpoints", 30, ch='u2add50/', ver=4)
    # opttest_comp_use_edge("Er", 100, "")
    
    # opttest_comp_use_edge("GMM12Sig8LCB4/checkpoints", 30, ch='ch100/', ver=4, edge_ver = 3)
    # opttest_comp_use_edge("GMM12Sig8LCB4_def/checkpoints", 30, ch='u1add50/', ver=4, edge_ver = 3)
    # opttest_comp_use_edge("GMM12Sig8LCB4_def/checkpoints", 30, ch='u2add50/', ver=4, edge_ver = 3)
    
    # opttest_comp_use_edge("GMM12Sig8LCB4/checkpoints", 30, ch='ch100/', ver=4, edge_ver = None)
    # opttest_comp_use_edge("GMM12Sig8LCB4_def/checkpoints", 30, ch='u2add50/', ver=4, edge_ver = None)
    
    # opttest_comp_use_edge("GMM12Sig8LCB4/checkpoints", 12, ch='u1add50/', ver=5, edge_ver = 3)
    
    # torch_test()
    
    pass