#coding: UTF-8
from ..learn import *

def Help():
    pass


def Run(ct,*args):
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "onpolicy/GMM4Sig005_gnnsd1_ggmm1_LCB2/t{}".format(args[0])
    num_ep = 500
    n_rand_sample = 10
    n_learn_step = 1
    max_smsz = 0.8
    nn_options = {
        'n_units': [2] + [200, 200] + [1],
        'n_units_err': [2] + [200, 200] + [1],
        # 'loss_stddev_stop_err': 1.0e-4,
        'error_loss_neg_weight': 0.1,
        # 'loss_stddev_stop': 1.0e-6,
        'num_check_stop': 50,
        "batchsize": 10,
    }
    n_save_ep = num_ep
    
    fixed_input = [
        (),
    ]
    
    use_gmm = True
    gmm_lam = lambda nnmodel: GMM4(nnmodel, diag_sigma=[(1.0-0.1)/20., (0.8-0.3)/20.], Gerr = 1.0)
    gain_pairs = (1.0, 1.0)
    # unobs = UnobservedSD(penalty = 0.02, diag_sigma=[(1.0-0.1)/100., (0.8-0.3)/100.])
    unobs = None
    LCB_ratio = 2.0
    
    logdir = base_logdir + "logs/{}/".format(name)
    modeldir = logdir + "{}/".format("models")
    
    if os.path.exists(logdir+"dm.pickle"):
        dm = Domain.load(logdir+"dm.pickle")
        dm.logdir = logdir
        dm.nnmodel.modeldir = modeldir
        dm.nnmodel.nn_options = nn_options
        dm.nnmodel.setup()
    else:
        nnmodel = NNModel(modeldir, nn_options)
        nnmodel.setup()
        # gmm = GMM(nnmodel, diag_sigma=[(1.0-0.1)/50, (0.8-0.3)/50], Gerr = Gerr)
        gmm = gmm_lam(nnmodel)
        dm = Domain(nnmodel, gmm, logdir, use_gmm = use_gmm, unobs = unobs, LCB_ratio = LCB_ratio, gain_pairs = gain_pairs)
        dm.setup()
        
    
    while len(dm.log["ep"]) < num_ep:
        dm.execute(n_rand_sample = n_rand_sample, n_learn_step = n_learn_step, max_smsz = max_smsz)
        if len(dm.log["ep"])%n_save_ep==0:
            dm.save()