#coding: UTF-8
from ..learn3 import *

class Domain4(Domain3, object):
    def __init__(self, logdir, sd_gain = 1.0, LCB_ratio = 0.0, without_smsz = None):
        super(Domain4, self).__init__(logdir, sd_gain, LCB_ratio, without_smsz)
        self.remain_idxs = {TIP: range(0,10000), SHAKE: range(0,100)}
                    
    def execute(self, num_tip, num_shake):
        ep = len(self.log["ep"])
        print("ep: {}".format(ep))
            
        flag = True
        if ep <= num_tip:
            skill = TIP
            idx_smsz_est_optparam = RandI(len(self.remain_idxs[TIP]))
            self.remain_idxs[TIP].pop(idx_smsz_est_optparam)
            idx_smsz = idx_smsz_est_optparam%100
            idx_est_optparam = idx_smsz_est_optparam/100
            smsz = self.smsz[idx_smsz]
            est_optparam = self.dtheta2[idx_est_optparam]
            not_learn = False if num_tip - ep <= 50 else True
        else:
            skill = SHAKE
            idx_smsz = RandI(len(self.remain_idxs[SHAKE]))
            self.remain_idxs[SHAKE].pop(idx_smsz)
            if ep >= num_tip + num_shake:
                flag = False
            smsz = self.smsz[idx_smsz]
            est_optparam = np.array(np.nan)
            not_learn = False if (num_tip + num_shake) - ep <= 50 else True
        est_datotal = 0
        opteval = 0
            
        if skill == TIP:    
            true_datotal = self.datotal[TIP][TRUE][idx_est_optparam, idx_smsz]
            self.nnmodels[TIP].update([est_optparam, smsz], [true_datotal], not_learn = not_learn)
        else:               
            true_datotal = self.datotal[SHAKE][TRUE][idx_smsz]
            self.nnmodels[SHAKE].update([smsz], [true_datotal], not_learn = not_learn)
        true_r_at_est_optparam = rfunc(true_datotal)
        
            
        self.log["ep"].append(ep)
        self.log["smsz"].append(smsz.item())
        self.log["skill"].append(skill)
        self.log["opteval"].append(opteval)
        self.log["est_optparam"].append(est_optparam.item())
        self.log["est_datotal_at_est_optparam"].append(est_datotal)
        self.log["datotal_at_est_optparam"].append(true_datotal.item())
        self.log["r_at_est_optparam"].append(true_r_at_est_optparam.item())
        if self.use_gmm:
            for skill in [TIP, SHAKE]:
                self.log["est_gmm_JP_{}".format(skill)].append(deepcopy(self.gmms[skill].jumppoints))
                
        return flag


def execute_checkpoint2(base_logdir, sd_gain, LCB_ratio, without_smsz, gmm_lams, num_tip, num_shake, num_checkpoints):
    num_ep = num_tip + num_shake
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
            dm = Domain4(new_logdir, sd_gain, LCB_ratio, without_smsz)
        dm.setup(gmm_lams)

        flag = True
        while flag:
            flag = dm.execute(num_tip, num_shake)
        dm.save()


def Run(ct,*args):
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "400_20/t{}".format(args[0])
    # p = 8
    # options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0}
    
    execute_checkpoint2(**dict(
        num_tip = 400,
        num_shake = 20,
        num_checkpoints= 1,
        
        without_smsz = None,
        
        sd_gain = 1.0,
        LCB_ratio = 0.0,
        # gmm_lams = {
        #     TIP: lambda nnmodel: GMM12(nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], w_positive = True, options = options, Gerr = 1.0),
        #     SHAKE: lambda nnmodel: GMM12(nnmodel, diag_sigma=[(0.8-0.3)/(100./p)], w_positive = True, options = options, Gerr = 1.0)
        # },
        gmm_lams = None,
        
        base_logdir = base_logdir + "logs/offpolicy/{}/".format(name),
    ))