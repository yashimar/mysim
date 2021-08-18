#coding: UTF-8
from ..learn3 import *

def Help():
    pass


def Run(ct,*args):
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "ErLCB4/checkpoints/t{}".format(args[0])    
    
    execute_checkpoint(**dict(
        num_ep = 500,
        num_rand_sample = 10,
        num_learn_step = 1,
        num_checkpoints= 1,
        
        sd_gain = 1.0,
        LCB_ratio = 4.0,
        gmm_lams = None,
        
        base_logdir = base_logdir + "logs/onpolicy2/{}/".format(name),
    ))