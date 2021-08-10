#coding: UTF-8
from .learn2 import *


TIP = "tip"
SHAKE = "shake"


class GMM7(GMM5, object):
    def train(self, recreate_jp = True): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        if recreate_jp:
            self.extract_jps()
        self.gc_concat = []
        self.w_concat = []
        Var = np.diag(self.diag_sigma)**2
        for jpx, jpy in zip(np.array(self.jumppoints["X"]), np.array(self.jumppoints["Y"])):
            self.gc_concat.append(
                lambda x,jpx=jpx,jpy=jpy: multivariate_normal.pdf(x,jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy
            )
        y = np.array(self.jumppoints["Y"]).flatten()
        X = np.array([[gc(x).item() for gc in self.gc_concat] for x in self.jumppoints["X"]])
        if len(X) == 0:
            self.w_concat = []
        else:
            self.w_concat, rnorm = nnls(X, y)


class Domain3:
    def __init__(self, logdir, sd_gain = 1.0, LCB_ratio = 0.0):
        self.smsz = np.linspace(0.3,0.8,100)
        self.dtheta2 = np.linspace(0.1,1,100)[::-1]
        self.datotal = {
            TIP: {TRUE: np.ones((100,100))*(-100), RFUNC: np.ones((100,100))*(-100)},
            SHAKE: {TRUE: np.ones(100)*(-100), RFUNC: np.ones(100)*(-100)}  #size_srcmouth のみ変動
        }
        self.nnmodels = dict()
        self.gmms = dict()
        self.use_gmm = False
        self.sd_gain = sd_gain
        self.LCB_ratio = LCB_ratio
        self.log = {
            "ep": [],
            "smsz": [],
            "skill": [],
            "opteval": [],
            "est_optparam": [],
            "est_datotal_at_est_optparam": [],
            "datotal_at_est_optparam": [],
            "r_at_est_optparam": [],
            # "est_gmm_JP_tip": [],
            # "est_gmm_JP_shake": [],
        }
        self.logdir = logdir
        self.rmodel = Rmodel("Fdatotal_gentle")

    def setup(self, gmm_lams = None):
        self.datotal[TIP][TRUE] = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/npdata/datotal.npy")
        self.datotal[SHAKE][TRUE] = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal.npy")
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
            self.log.update({
                "est_gmm_JP_tip": [],
                "est_gmm_JP_shake": [],
            })
            for skill in [TIP, SHAKE]:
                self.gmms[skill] = gmm_lams[skill](self.nnmodels[skill])
                    
    def optimize(self, smsz):
        ###########################
        ###Tip用最適化
        ###########################
        est_nn_Er, est_nn_Sr = [], []
        if self.use_gmm:
            self.gmms[TIP].train()
            X = np.array([[dtheta2, smsz] for dtheta2 in self.dtheta2])
            SDgmm = self.gmms[TIP].predict(X)
        for idx_dtheta2, dtheta2 in enumerate(self.dtheta2):
            x_in = [dtheta2, smsz]
            est_datotal = self.nnmodels[TIP].model.Predict(x = x_in, with_var=True)
            if self.use_gmm:
                x_var = [0, (self.sd_gain*(math.sqrt(est_datotal.Var[0,0].item()) + SDgmm[idx_dtheta2].item()))**2]
            else:
                x_var = [0, (self.sd_gain*math.sqrt(est_datotal.Var[0].item()))**2]
            est_nn = self.rmodel.Predict(x=[0.3, est_datotal.Y[0].item()], x_var= x_var, with_var=True)
            est_nn_Er.append(est_nn.Y.item())
            est_nn_Sr.append(np.sqrt(est_nn.Var[0,0]).item())    
        evalmatrix = np.array(est_nn_Er) - self.LCB_ratio*np.array(est_nn_Sr)
        idx_est_optparam = np.argmax(evalmatrix)
        est_optparam = self.dtheta2[idx_est_optparam]
        opteval_TIP = est_nn_Er[idx_est_optparam]
        
        ###########################
        ###Shake用最適化
        ###########################
        if self.use_gmm:
            self.gmms[SHAKE].train()
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
            est_datotal = self.nnmodels[SHAKE].model.Predict(x = [smsz], with_var=True)
            opteval = opteval_SHAKE
        else:
            skill = TIP
            est_datotal = self.nnmodels[TIP].model.Predict(x=[est_optparam, smsz], with_var=True).Y[0].item()
            opteval = opteval_TIP
        
        return skill, idx_est_optparam, est_optparam, est_datotal, opteval
                    
    def execute_main(self, idx_smsz, smsz, n_rand_sample, n_learn_step):
        ep = len(self.log["ep"])
        print("ep: {}".format(ep))
            
        if ep<= n_rand_sample:
            if ep <= int(n_rand_sample/2):    #Tip用ランダムサンプリング
                skill = TIP
                idx_est_optparam = RandI(len(self.dtheta2))
                est_optparam = self.dtheta2[idx_est_optparam]
            else:   #Shake用ランダムサンプリング
                skill = SHAKE
                idx_est_optparam = np.nan
                est_optparam = np.array(np.nan)
            est_datotal = 0
            opteval = 0
        else:
            skill, idx_est_optparam, est_optparam, est_datotal, opteval = self.optimize(smsz)
            
        if skill == TIP:    true_datotal = self.datotal[TIP][TRUE][idx_est_optparam, idx_smsz]
        else:               true_datotal = self.datotal[SHAKE][TRUE][idx_smsz]
        true_r_at_est_optparam = rfunc(true_datotal)
        
        if ep % n_learn_step == 0:  not_learn = False
        else:                       not_learn = True
        
        if skill == TIP:    self.nnmodels[TIP].update([est_optparam, smsz], [true_datotal], not_learn = not_learn)
        else:               self.nnmodels[SHAKE].update([smsz], [true_datotal], not_learn = not_learn)
            
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
        
    def execute(self, n_rand_sample, n_learn_step):
        idx_smsz = RandI(len(self.smsz))
        smsz = self.smsz[idx_smsz]
        self.execute_main(idx_smsz, smsz, n_rand_sample, n_learn_step)
            
    @classmethod
    def load(self, path):
        with open(path, mode="rb") as f:
            dm = dill.load(f)
        
        return dm
    
    def save(self):
        with open(self.logdir+"log.yaml", "w") as f:
            yaml.dump(self.log, f)
        with open(self.logdir+"dm.pickle", mode="wb") as f:
            dill.dump(self, f)
        

def shake_rfunc_plot():
    datotal = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal.npy")
    r = []
    for d in datotal:
        r.append(rfunc(d))
    plt.scatter(x = np.linspace(0.3,0.8,100), y = r)


def execute():
    pass       
        
def test():
    #Domain用パラメータ
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "test"
    num_ep = 20
    n_rand_sample = 10
    
    #NN学習用パラメータ
    n_learn_step = 1
    
    #実験イテレーション用パラメータ
    n_save_ep = num_ep
    
    #最適化用パラメータ
    p = 3
    gmm_lams = {
        TIP: lambda nnmodel: GMM7(nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], Gerr = 1.0),
        SHAKE: lambda nnmodel: GMM7(nnmodel, diag_sigma=[(0.8-0.3)/(100./p)], Gerr = 1.0)
    }
    sd_gain = 1.0
    LCB_ratio = 0.0

    
    logdir = base_logdir + "logs/onpolicy2/{}/".format(name)
    modeldir = logdir + "{}/".format("models")
    
    if os.path.exists(logdir+"dm.pickle"):
        dm = Domain.load(logdir+"dm.pickle")
    else:
        dm = Domain3(logdir, sd_gain, LCB_ratio)
        dm.setup(gmm_lams)
    
    while len(dm.log["ep"]) < num_ep:
        dm.execute(n_rand_sample = n_rand_sample, n_learn_step = n_learn_step)
        if len(dm.log["ep"])%n_save_ep==0:
            dm.save()

        
def Run(ct, *args):
    test()
    