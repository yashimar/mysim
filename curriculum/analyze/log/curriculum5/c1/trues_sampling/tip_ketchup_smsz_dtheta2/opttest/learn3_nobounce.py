from .learn3 import *

class Domain3_nb(Domain3, object):
    def setup(self, gmm_lams = None):
        self.datotal[TIP][TRUE] = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_nobounce_smsz_dtheta2/npdata/datotal.npy")
        self.datotal[SHAKE][TRUE] = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/shake_nobounce_smsz/npdata/datotal.npy")
        for i in range(100):
            self.datotal[SHAKE][RFUNC][i] = rfunc(self.datotal[SHAKE][TRUE][i].item())
            for j in range(100):
                self.datotal[TIP][RFUNC][i,j] = rfunc(self.datotal[TIP][TRUE][i,j].item())
        
        nn_options = dict()
        nn_options[TIP] = {
            'n_units': [2] + [200, 200] + [1],
            'n_units_err': [2] + [200, 200] + [1],
            'error_loss_neg_weight': 0.1,
            'num_check_stop': 50,
            "batchsize": 10,
        }
        nn_options[SHAKE] = {
            'n_units': [1] + [200, 200] + [1],
            'n_units_err': [1] + [200, 200] + [1],
            'error_loss_neg_weight': 0.1,
            'num_check_stop': 50,
            "batchsize": 10,
        }
        for skill in [TIP, SHAKE]:
            modeldir = self.logdir + "models/{}/".format(skill)
            nnmodel = NNModel(modeldir, nn_options[skill])
            nnmodel.setup()
            self.nnmodels[skill] = nnmodel
            
        if gmm_lams != None:
            self.use_gmm = True
            for skill in [TIP, SHAKE]:
                self.gmms[skill] = gmm_lams[skill](self.nnmodels[skill])
                
                
def execute(logdir, sd_gain, LCB_ratio, gmm_lams, num_ep, num_rand_sample, num_learn_step, num_save_ep):
    if os.path.exists(logdir+"dm.pickle"):
        dm = Domain3_nb.load(logdir+"dm.pickle")
    else:
        dm = Domain3_nb(logdir, sd_gain, LCB_ratio)
        dm.setup(gmm_lams)
    
    while len(dm.log["ep"]) < num_ep:
        dm.execute(num_rand_sample = num_rand_sample, num_learn_step = num_learn_step)
        if len(dm.log["ep"])%num_save_ep==0:
            dm.save()
        

def Run(ct, *args):
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_nobounce_smsz_dtheta2/opttest/"
    name = "Er/t{}".format(args[0])    
    
    execute(**dict(
        num_ep = 500,
        num_rand_sample = 20,
        num_learn_step = 1,
        num_save_ep = 500,
        
        sd_gain = 1.0,
        LCB_ratio = 0.0,
        gmm_lams = None,
        
        logdir = base_logdir + "logs/onpolicy2/{}/".format(name),
    ))