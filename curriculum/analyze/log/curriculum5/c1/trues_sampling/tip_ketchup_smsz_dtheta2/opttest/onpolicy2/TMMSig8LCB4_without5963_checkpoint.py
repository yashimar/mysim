#coding: UTF-8
from ..learn3 import *

def Help():
    pass


def Run(ct,*args):
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "TMMSig8LCB4_without5963/checkpoints/t{}".format(args[0])
    p = 8
    options = {"tau": 0.9, "lam": 1e-6}
    
    execute_checkpoint(**dict(
        num_ep = 500,
        num_rand_sample = 10,
        num_learn_step = 1,
        num_checkpoints= 20,
        
        without_smsz = (0.59, 0.63),
        
        sd_gain = 1.0,
        LCB_ratio = 4.,
        gmm_lams = {
            TIP: lambda nnmodel: TMM(nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], options = options, Gerr = 1.0),
            SHAKE: lambda nnmodel: TMM(nnmodel, diag_sigma=[(0.8-0.3)/(100./p)], options = options, Gerr = 1.0)
        },
        
        base_logdir = base_logdir + "logs/onpolicy2/{}/".format(name),
    ))