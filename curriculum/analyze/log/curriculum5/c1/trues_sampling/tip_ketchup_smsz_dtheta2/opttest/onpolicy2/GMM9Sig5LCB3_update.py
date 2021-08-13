#coding: UTF-8
from ..learn3 import *

def Help():
    pass


def Run(ct,*args):
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "GMM9Sig5LCB3/t{}".format(args[0])
    p = 5
    options = {"tau": 0.9, "lam": 1e-6}
        
    execute_update(**dict(
        num_ep = 1000,
        num_rand_sample = 10,
        num_learn_step = 1,
        num_save_ep = 500,
        
        gmm_lams = {
            TIP: lambda nnmodel: GMM9(nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], options = options, Gerr = 1.0),
            SHAKE: lambda nnmodel: GMM9(nnmodel, diag_sigma=[(0.8-0.3)/(100./p)], options = options, Gerr = 1.0)
        },
        
        ref_logdir = base_logdir + "logs/onpolicy2/{}/".format(name),
        new_logdir = base_logdir + "logs/onpolicy2/{}/update1000/".format(name),
    ))