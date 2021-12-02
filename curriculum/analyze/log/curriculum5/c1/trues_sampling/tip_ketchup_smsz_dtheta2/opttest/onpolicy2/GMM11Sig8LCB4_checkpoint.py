#coding: UTF-8
from ..learn3 import *

def Help():
    pass


def Run(ct,*args):
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "GMM11Sig8LCB4/checkpoints/t{}".format(args[0])
    p = 8
    options = {"tau": 0.9, "lam": 0, 'maxiter': 1e2, 'popsize': 100, 'verbose': 0}
    
    execute_checkpoint(**dict(
        num_ep = 500,
        num_rand_sample = 10,
        num_learn_step = 1,
        num_checkpoints= 1,
        
        without_smsz = None,
        
        sd_gain = 1.0,
        LCB_ratio = 4.0,
        gmm_lams = {
            TIP: lambda nnmodel: GMM11(nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], w_positive = True, options = options, Gerr = 1.0),
            SHAKE: lambda nnmodel: GMM11(nnmodel, diag_sigma=[(0.8-0.3)/(100./p)], w_positive = True, options = options, Gerr = 1.0)
        },
        
        base_logdir = base_logdir + "logs/onpolicy2/{}/".format(name),
    ))