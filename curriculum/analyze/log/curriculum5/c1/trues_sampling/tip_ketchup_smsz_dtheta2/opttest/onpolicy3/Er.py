#coding: UTF-8
from ..learn3 import *

def Help():
    pass


def Run(ct,*args):
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "Er/t{}".format(args[0])    
    
    execute(**dict(
        num_ep = 500,
        num_rand_sample = 10,
        num_learn_step = 1,
        num_save_ep = 500,
        
        sd_gain = 1.0,
        LCB_ratio = 0.0,
        gmm_lams = None,
        
        logdir = base_logdir + "logs/onpolicy3/{}/".format(name),
    ))