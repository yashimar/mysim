from learn3 import *


class Domain4(Domain3, object):
    def __init__(self, logdir, sd_gain = 1.0, LCB_ratio = 0.0, without_smsz = None, start_smsz_select = 100, eval_min_thr = -5):
        super(Domain4, self).__init__(logdir, sd_gain = sd_gain, LCB_ratio = LCB_ratio, without_smsz = without_smsz)
        self.start_smsz_select = start_smsz_select
        self.eval_min_thr = eval_min_thr
    
    
    def select_smsz_idx_from_eval_tip(self):
        eval_min_thr = self.eval_min_thr
        
        model = self.nnmodels[TIP].model
        X = np.array([[dtheta2, smsz] for dtheta2 in self.dtheta2 for smsz in self.smsz ]).astype(np.float32)
        datotal_nnmean = model.Forward(x_data = X, train = False).data.reshape(100,100)
        datotal_nnsd = np.array([np.sqrt(model.Predict(x = [dtheta2, smsz], with_var = True).Var[0,0].item()) for dtheta2 in self.dtheta2 for smsz in self.smsz]).reshape(100,100)
        if self.use_gmm:
            SDgmm = self.gmms[TIP].predict(X).reshape((100,100))
        else:
            SDgmm = np.zeros(100,100)
        rnn_sm = np.array([[self.rmodel.Predict(x=[0.3, datotal_nnmean[idx_dtheta2, idx_smsz]], x_var=[0, (self.sd_gain*(datotal_nnsd[idx_dtheta2, idx_smsz] + SDgmm[idx_dtheta2, idx_smsz]))**2], with_var=True) for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        tip_Er = np.array([[rnn_sm[idx_dtheta2, idx_smsz].Y.item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        tip_Sr = np.sqrt([[rnn_sm[idx_dtheta2, idx_smsz].Var[0,0].item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        eval_tip = tip_Er - self.LCB_ratio*tip_Sr
            
        eval_tip = np.maximum(eval_tip, eval_min_thr)
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
        fitted_curve = interpolate.interp1d(self.smsz, y, kind='cubic')
        fy = np.maximum(fitted_curve(self.smsz), 0)
        idx_smsz = np.random.choice(range(len(fy)), p=fy/np.sum(fy))
        
        return idx_smsz
    
        
    def execute(self, num_rand_sample, num_learn_step):
        if len(self.log["ep"]) >= self.start_smsz_select:
            idx_smsz = self.select_smsz_idx_from_eval_tip()
        else:
            idx_smsz = RandI(len(self.smsz))
        # ep = len(self.log["ep"])
        # idx_smsz = ep%100
        smsz = self.smsz[idx_smsz]
        if self.without_smsz != None:
            while (self.without_smsz[0] < smsz.item() < self.without_smsz[1]):
                idx_smsz = RandI(len(self.smsz))
                smsz = self.smsz[idx_smsz]
        self.execute_main(idx_smsz, smsz, num_rand_sample, num_learn_step)


def execute_checkpoint2(base_logdir, sd_gain, LCB_ratio, without_smsz, gmm_lams, num_ep, num_rand_sample, num_learn_step, num_checkpoints, start_smsz_select, eval_min_thr):
    ep_checkpoints = [num_ep/num_checkpoints*i for i in range(1,num_checkpoints+1)]
    for ep_checkpoint in ep_checkpoints:
        new_logdir = base_logdir + "ch{}/".format(ep_checkpoint)
        prev_logdir = base_logdir + "ch{}/".format(ep_checkpoint - num_ep/num_checkpoints)
        os.makedirs(new_logdir)
        if os.path.exists(prev_logdir+"dm.pickle"):
            shutil.copytree(prev_logdir+"models", new_logdir+"models")
            dm = Domain4.load(prev_logdir+"dm.pickle")
            dm.logdir = new_logdir
        else:
            dm = Domain4(new_logdir, sd_gain, LCB_ratio, without_smsz, start_smsz_select = start_smsz_select, eval_min_thr = eval_min_thr)
        dm.setup(gmm_lams)
        
        while len(dm.log["ep"]) < ep_checkpoint:
            dm.execute(num_rand_sample = num_rand_sample, num_learn_step = num_learn_step)
        dm.save()
        
        
def execute_update(ref_logdir, new_logdir, gmm_lams, num_ep, num_rand_sample, num_learn_step, start_smsz_select = None):
    os.makedirs(new_logdir)
    shutil.copytree(ref_logdir+"models", new_logdir+"models")
    dm = Domain4.load(ref_logdir+"dm.pickle")
    dm.logdir = new_logdir
    dm.setup(gmm_lams)
    if start_smsz_select != None:
        dm.start_smsz_select = start_smsz_select
    
    init_ep = len(dm.log["ep"])
    for _ in range(init_ep,init_ep+num_ep):
        dm.execute(num_rand_sample = num_rand_sample, num_learn_step = num_learn_step)
    dm.save()
    

def hist_concat3(ver=4):
    n = 12
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    # name = "Er"
    # name = "GMM12Sig8LCB4/checkpoints"
    names = [
        # ("GMM12Sig8LCB4/checkpoints", "ch100/"),
        # ("GMM12Sig8LCB4/checkpoints", "u3add50/"),
        # ("GMM12Sig8LCB4_def/checkpoints", "u2add50/"),
        ("GMM12Sig8LCB4/checkpoints", "u1add50/"),
    ]
    vis_names = [
        # "curriculum",
        # 'normal',
        'use edge'
    ]
    colors = [
        # 'red',
        "blue",
    ]
    w_size = 1
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/hist_concat/".format(ver)
    check_or_create_dir(save_img_dir)
    
    fig = go.Figure()
    r_hist_mean_concat = []
    total_ep = 100
    for n_i, (name, ch) in enumerate(names):
        r_hist_concat = []
        # rng = range(1,100) if n_i != 2 else range(2,30)+range(90,100)
        rng = range(1,n)
        for i in rng:
            logdir = basedir +"{}/t{}/{}".format(name, i, ch)
            print(logdir)
            # with open(logdir+"log.yaml", "r") as yml:
            #     log = yaml.load(yml)
            dm = Domain4.load(logdir+"dm.pickle")
            r_hist_concat.append(dm.log["r_at_est_optparam"])
        # r_hist_mean = np.mean(r_hist_concat, axis = 0)
        # r_hist_sd = np.std(r_hist_concat, axis = 0)
        r_hist_mean = [np.mean(np.array(r_hist_concat)[:,i-w_size:i]) for i in range(w_size,total_ep)]
        r_hist_sd = [np.std(np.array(r_hist_concat)[:,i-w_size:i]) for i in range(w_size,total_ep)]
        r_hist_p50 = [np.percentile(np.array(r_hist_concat)[:,i-w_size:i], 50) for i in range(w_size,total_ep)]
        r_hist_pmin = [np.percentile(np.array(r_hist_concat)[:,i-w_size:i], 2) for i in range(w_size,total_ep)]
        r_hist_pmax = [np.percentile(np.array(r_hist_concat)[:,i-w_size:i], 98) for i in range(w_size,total_ep)]
        r_hist_mean_concat.append(r_hist_mean)
        
    for n_i, (name, ch) in enumerate(names):
        r_hist_mean = r_hist_mean_concat[n_i]
        fig.add_trace(go.Scatter(
            # name = name,
            name = vis_names[n_i],
            x = dm.log["ep"][w_size:], 
            y = r_hist_mean,
            # y = r_hist_p50,
            line = dict(color = colors[n_i], width = 4),
            # error_y = dict(
            #     type ="data",
            #     symmetric = True,
            #     array = r_hist_sd,
            #     # symmetric = False,
            #     # array = np.array(r_hist_pmax) - np.array(r_hist_p50),
            #     # arrayminus = np.array(r_hist_p50)-np.array(r_hist_pmin),
            #     # arrayminus = np.array(r_hist_p50)-np.array(r_hist_pmin),
            #     thickness = 1.5,
            #     width = 3,
            # ),
            # mode = "markers",
            mode = "lines",
        ))
    fig.add_shape(type="line",
        x0=10, y0=-3, x1=10, y1=0.2,
        line=dict(
            color="black",
            width=3,
            dash="dash",
    ))
    fig.add_shape(type="line",
        x0=20, y0=-3, x1=20, y1=0.2,
        line=dict(
            color="black",
            width=3,
            dash="dash",
    ))
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        width = 1600,
        height = 900,
    )
    fig['layout']['xaxis']['range'] = (-2,502)
    fig['layout']['yaxis']['range'] = (-3,0.2)
    fig['layout']['xaxis']['title'] = "Episode"
    fig['layout']['yaxis']['title'] = "Average reward"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    plotly.offline.plot(fig, filename = save_img_dir + "{}.html".format(w_size), auto_open=False)
    fig.write_image(save_img_dir + "{}.svg".format(w_size))


def Run(ct, *args):
    # hist_concat3()
    hist_concat3(ver = 5)