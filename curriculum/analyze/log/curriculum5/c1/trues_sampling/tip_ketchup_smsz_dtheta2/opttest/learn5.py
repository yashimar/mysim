#coding: UTF-8
from mode_edge import *


class Domain5(Domain3, object):
    def optimize(self, smsz):
        ###########################
        ###Tip用最適化
        ###########################
        # est_nn_Er, est_nn_Sr = [], []
        # if self.use_gmm:
        #     # self.gmms[TIP].train()
        #     X = np.array([[dtheta2, smsz] for dtheta2 in self.dtheta2])
        #     SDgmm = self.gmms[TIP].predict(X)
        # for idx_dtheta2, dtheta2 in enumerate(self.dtheta2):
        #     x_in = [dtheta2, smsz]
        #     est_datotal = self.nnmodels[TIP].model.Predict(x = x_in, with_var=True)
        #     if self.use_gmm:
        #         x_var = [0, (self.sd_gain*(math.sqrt(est_datotal.Var[0,0].item()) + SDgmm[idx_dtheta2].item()))**2]
        #     else:
        #         x_var = [0, (self.sd_gain*math.sqrt(est_datotal.Var[0].item()))**2]
        #     est_nn = self.rmodel.Predict(x=[0.3, est_datotal.Y[0].item()], x_var= x_var, with_var=True)
        #     est_nn_Er.append(est_nn.Y.item())
        #     est_nn_Sr.append(np.sqrt(est_nn.Var[0,0]).item())
        
        idx_smsz = idx_of_the_nearest(self.smsz, smsz)
        
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
            
        pred, pred_edge, pred_edge2, pred_edge2_binary, fixed_eval, X_train, y_train = detect_edge3(self, eval_tip)
        evalmatrix = fixed_eval[idx_smsz]
        idx_est_optparam = np.argmax(evalmatrix)
        est_optparam = self.dtheta2[idx_est_optparam]
        opteval_TIP = evalmatrix[idx_est_optparam]
        
        ###########################
        ###Shake用最適化
        ###########################
        if self.use_gmm:
            # self.gmms[SHAKE].train()
            X = np.array([[smsz]])
            SDgmm = self.gmms[SHAKE].predict(X)
        est_datotal = self.nnmodels[SHAKE].model.Predict(x = [smsz], with_var=True)
        if self.use_gmm:
            x_var = [0, (self.sd_gain*(math.sqrt(est_datotal.Var[0,0].item()) + SDgmm[0].item()))**2]
        else:
            x_var = [0, (self.sd_gain*math.sqrt(est_datotal.Var[0].item()))**2]
        est_nn = self.rmodel.Predict(x=[0.3, est_datotal.Y[0].item()], x_var= x_var, with_var=True)
        opteval_SHAKE = est_nn.Y.item() - self.LCB_ratio*np.sqrt(est_nn.Var[0,0]).item()
        
        if opteval_SHAKE > opteval_TIP:
            skill  = SHAKE
            idx_est_optparam = np.nan
            est_optparm = np.nan
            est_datotal = self.nnmodels[SHAKE].model.Predict(x = [smsz], with_var=True).Y[0].item()
            opteval = opteval_SHAKE
        else:
            skill = TIP
            est_datotal = self.nnmodels[TIP].model.Predict(x=[est_optparam, smsz], with_var=True).Y[0].item()
            opteval = opteval_TIP
            
        if self.use_gmm:
            self.gmms[TIP].train()
            self.gmms[SHAKE].train()
        
        return skill, idx_est_optparam, est_optparam, est_datotal, opteval
    
    
def execute_checkpoint(base_logdir, sd_gain, LCB_ratio, without_smsz, gmm_lams, num_ep, num_rand_sample, num_learn_step, num_checkpoints, ver = 5):
    ep_checkpoints = [num_ep/num_checkpoints*i for i in range(1,num_checkpoints+1)]
    for ep_checkpoint in ep_checkpoints:
        new_logdir = base_logdir + "ch{}/".format(ep_checkpoint)
        prev_logdir = base_logdir + "ch{}/".format(ep_checkpoint - num_ep/num_checkpoints)
        os.makedirs(new_logdir)
        if os.path.exists(prev_logdir+"dm.pickle"):
            shutil.copytree(prev_logdir+"models", new_logdir+"models")
            if ver == 5:
                dm = Domain5.load(prev_logdir+"dm.pickle")
            dm.logdir = new_logdir
        else:
            if ver == 5:
                dm = Domain5(new_logdir, sd_gain, LCB_ratio, without_smsz)
        dm.setup(gmm_lams)
        
        while len(dm.log["ep"]) < ep_checkpoint:
            dm.execute(num_rand_sample = num_rand_sample, num_learn_step = num_learn_step)
        dm.save()
        
        
def execute_update(ref_logdir, new_logdir, gmm_lams, num_ep, num_rand_sample, num_learn_step, ver = 5):
    os.makedirs(new_logdir)
    shutil.copytree(ref_logdir+"models", new_logdir+"models")
    if ver == 5:
        dm = Domain5.load(ref_logdir+"dm.pickle")
    dm.logdir = new_logdir
    dm.setup(gmm_lams)
    
    init_ep = len(dm.log["ep"])
    for _ in range(init_ep,init_ep+num_ep):
        dm.execute(num_rand_sample = num_rand_sample, num_learn_step = num_learn_step)
    dm.save()