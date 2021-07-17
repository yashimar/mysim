#coding: UTF-8
from core_tool import *
from ay_py.core import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from scipy.stats import multivariate_normal
from .......util import *
from ........tasks_domain.util import Rmodel
from ..greedyopt import *
from copy import deepcopy
import dill
import math


RFUNC = "r_func"


def Help():
    pass


def idx_of_the_nearest(data, value):
    idx = np.argmin(np.abs(np.array(data) - value))
    return idx


class Domain:
    def __init__(self, nnmodel, gmm, logdir, use_gmm = False, LCB_ratio = 0.0, gain_pairs = None):
        self.smsz = np.linspace(0.3,0.8,100)
        self.dtheta2 = np.linspace(0.1,1,100)[::-1]
        self.datotal = {TRUE: np.ones((100,100))*(-100), RFUNC: np.ones((100,100))*(-100)}
        self.nnmodel = nnmodel
        self.gmm = gmm
        self.use_gmm = use_gmm
        self.LCB_ratio = LCB_ratio
        self.log = {
            "ep": [],
            "smsz": [],
            "est_opt_dtheta2": [],
            "true_opt_dtheta2": [],
            "est_datotal": [],
            "true_datotal": [],
            "est_opt_Er": [],
            "true_opt_r": [],
            "true_r_at_est_opt_dthtea2": [],
            "est_gmm_JP": [],
        }
        # self.logE = []
        self.logdir = logdir
        self.rmodel = Rmodel("Fdatotal_gentle")
        self.gain_pairs = gain_pairs if gain_pairs != None else (1.0, 1.0)

    def setup(self):
        self.datotal[TRUE] = np.load(SRC_PATH+"datotal.npy")
        for i in range(100):
            for j in range(100):
                self.datotal[RFUNC][i,j] = rfunc(self.datotal[TRUE][i,j].item())
    
    def optimize(self, smsz):
        est_nn_Er, est_nn_Sr = [], []
        for dtheta2 in self.dtheta2:
            x_in = [dtheta2, smsz]
            est_datotal = self.nnmodel.model.Predict(x = x_in, with_var=True)
            
            if self.use_gmm:
                self.gmm.train()
                # x_var = [0, max(est_datotal.Var[0].item(), self.gmm.predict(x_in).item()**2)]
                x_var = [0, (self.gain_pairs[0]*math.sqrt(est_datotal.Var[0,0].item()) + self.gain_pairs[1]*self.gmm.predict(x_in).item())**2]
            else:
                x_var = [0, est_datotal.Var[0].item()]
            est_nn = self.rmodel.Predict(x=[0.3, est_datotal.Y[0].item()], x_var= x_var, with_var=True)
            est_nn_Er.append(est_nn.Y.item())
            est_nn_Sr.append(np.sqrt(est_nn.Var[0,0]).item())
            
        E = np.array(est_nn_Er) - self.LCB_ratio*np.array(est_nn_Sr)
        idx_est_opt_dtheta2 = np.argmax(E)
        est_opt_dtheta2 = self.dtheta2[idx_est_opt_dtheta2]
        est_datotal = self.nnmodel.model.Predict(x=[est_opt_dtheta2, smsz], with_var=True).Y[0].item()
        est_opt_Er = est_nn_Er[idx_est_opt_dtheta2]
        
        return idx_est_opt_dtheta2, est_opt_dtheta2, est_datotal, est_opt_Er, E
                    
    def execute_main(self, idx_smsz, smsz, n_rand_sample, n_learn_step, fixed_input):
        ep = len(self.nnmodel.model.DataX)
        print("ep: {}".format(ep))
        true_opt_dtheta2_idx = np.argmax(self.datotal[RFUNC][:, idx_smsz])
        true_opt_dtheta2 = self.dtheta2[true_opt_dtheta2_idx]
        true_opt_r = self.datotal[RFUNC][true_opt_dtheta2_idx, idx_smsz]
        # self.gmm.train()
            
        if fixed_input != None:
            est_opt_dtheta2 = fixed_input[1]
            idx_est_opt_dtheta2 = idx_of_the_nearest(self.dtheta2, est_opt_dtheta2)
            est_datotal = 0
            est_opt_Er = 0
            E = np.zeros((len(self.dtheta2),len(self.smsz)))
        elif ep <= n_rand_sample:
            idx_est_opt_dtheta2 = RandI(len(self.dtheta2))
            est_opt_dtheta2 = self.dtheta2[idx_est_opt_dtheta2]
            est_datotal = 0
            est_opt_Er = 0
            E = np.zeros((len(self.dtheta2),len(self.smsz)))
        else:
            idx_est_opt_dtheta2, est_opt_dtheta2, est_datotal, est_opt_Er, E = self.optimize(smsz)
            
        true_datotal = self.datotal[TRUE][idx_est_opt_dtheta2, idx_smsz]
        true_r_at_est_opt_dthtea2 = rfunc(true_datotal)
            
        if ep % n_learn_step == 0:
            self.nnmodel.update([est_opt_dtheta2, smsz], [true_datotal], not_learn = False)
        else:
            self.nnmodel.update([est_opt_dtheta2, smsz], [true_datotal], not_learn = True)
            
        self.log["ep"].append(ep)
        self.log["smsz"].append(smsz.item())
        self.log["est_opt_dtheta2"].append(est_opt_dtheta2.item())
        self.log["true_opt_dtheta2"].append(true_opt_dtheta2.item())
        self.log["est_datotal"].append(est_datotal)
        self.log["true_datotal"].append(true_datotal.item())
        self.log["est_opt_Er"].append(est_opt_Er)
        self.log["true_opt_r"].append(true_opt_r.item())
        self.log["true_r_at_est_opt_dthtea2"].append(true_r_at_est_opt_dthtea2.item())
        self.log["est_gmm_JP"].append(deepcopy(self.gmm.jumppoints))
        # self.logE.append(E)
            
            
    def execute(self, n_rand_sample, n_learn_step, max_smsz = 0.8, fixed_input = None):
        idx_smsz = RandI(len(self.smsz))
        smsz = self.smsz[idx_smsz]
        while smsz > max_smsz:
            idx_smsz = RandI(len(self.smsz))
            smsz = self.smsz[idx_smsz]
        if fixed_input != None:
            smsz = fixed_input[0]
            idx_smsz = idx_of_the_nearest(self.smsz, smsz)
        self.execute_main(idx_smsz, smsz, n_rand_sample, n_learn_step, fixed_input)
            
    @classmethod
    def load(self, path):
        with open(path, mode="rb") as f:
            dm = dill.load(f)
        # E = pd.read_csv(self.logdir+"E.csv").to_numpy().tolist()
        # dm.logE = E
        
        return dm
    
    def save(self):
        with open(self.logdir+"log.yaml", "w") as f:
            yaml.dump(self.log, f)
        # np.save(self.logdir+"E.npy", np.array(self.logE))
        # print(np.array(self.logE).shape)
        # E = pd.DataFrame(data=np.array(self.logE))
        # E.to_csv(self.logdir+"E.csv")
        # self.logE = None
        
        with open(self.logdir+"dm.pickle", mode="wb") as f:
            dill.dump(self, f)
        
class NNModel:
    def __init__(self, modeldir, nn_options):
        self.basedir = modeldir
        self.nn_options = nn_options
        self.model = None
    
    def setup(self):
        dim_in = 2
        dim_out = 1
        options={
            'base_dir': self.basedir,
            'n_units': [dim_in] + [200, 200, 200] + [dim_out],
            'name': 'Fdatotal',
            'loss_stddev_stop': 1.0e-3,
            'loss_stddev_init': 2.0,
            'error_loss_neg_weight': 0.1,
            'num_max_update': 5000,
            "batchsize": 10,
            'num_check_stop': 50,
        }
        options.update(self.nn_options)
        self.model = TNNRegression()
        self.model.Load(data={'options':options})
        if os.path.exists(self.basedir+'setup.yaml'):
            self.model.Load(LoadYAML(self.basedir+'setup.yaml'), base_dir=self.basedir)
            self.model.Load(data={'options':options}, base_dir=self.basedir)
        self.model.Init()
        Print("len(DataX): ", len(self.model.DataX))
        check_or_create_dir(self.basedir)
        
    def update(self, x, y, not_learn = False):
        self.model.Update(x, y, not_learn)
        SaveYAML(self.model.Save(self.basedir), self.basedir+'setup.yaml')


class GMM:
    def __init__(self, nnmodel, diag_sigma, Gerr):
        self.nnmodel = nnmodel
        self.jumppoints = {"X": [], "Y": []}
        self.gaussian_components = []
        self.diag_sigma = diag_sigma
        self.Gerr = Gerr
    
    def extract_jps(self):
        self.jumppoints = {"X": [], "Y": []}
        model = self.nnmodel.model
        for i, (x, y) in enumerate(zip(model.DataX, model.DataY)):
            p_mean = model.Forward(x_data = model.DataX[i:i+1], train = False).data.item() #Chainerのバグ回避用
            p_err = model.ForwardErr(x_data = model.DataX[i:i+1], train = False).data.item()
            if (y < (p_mean - self.Gerr*p_err) or (y > (p_mean + self.Gerr*p_err))):
                self.jumppoints["X"].append(x.tolist())
                self.jumppoints["Y"].append(abs(y - p_mean).tolist())
                
    def train(self): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        self.extract_jps()
        Var = np.diag(self.diag_sigma)**2
        for jpx, jpy in zip(np.array(self.jumppoints["X"]), np.array(self.jumppoints["Y"])):
            self.gaussian_components.append(lambda x,jpx=jpx,jpy=jpy: multivariate_normal.pdf(x,jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy)
    
    def predict(self, x):
        pred = 0
        for gc in self.gaussian_components:
            pred += gc(x)
        return pred


class GMM2:
    def __init__(self, nnmodel, diag_sigma, Gerr):
        self.nnmodel = nnmodel
        self.jumppoints = {"X": [], "Y": []}
        self.gaussian_components = []
        self.diag_sigma = diag_sigma
        self.Gerr = Gerr
    
    def extract_jps(self):
        self.jumppoints = {"X": [], "Y": []}
        model = self.nnmodel.model
        p_means = model.Forward(x_data = model.DataX, train = False).data
        p_errs = model.ForwardErr(x_data = model.DataX, train = False).data
        for i, (p_mean, p_err, x, y) in enumerate(zip(p_means, p_errs, model.DataX, model.DataY)):
            if (y < (p_mean - self.Gerr*p_err) or (y > (p_mean + self.Gerr*p_err))):
                jp = max((p_mean - self.Gerr*p_err)-y, y-(p_mean + self.Gerr*p_err))
                self.jumppoints["X"].append(x.tolist())
                self.jumppoints["Y"].append(jp.tolist())
        
                
    def train(self): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        self.extract_jps()
        Var = np.diag(self.diag_sigma)**2
        for jpx, jpy in zip(np.array(self.jumppoints["X"]), np.array(self.jumppoints["Y"])):
            self.gaussian_components.append(lambda x,jpx=jpx,jpy=jpy: multivariate_normal.pdf(x,jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy)
    
    def predict(self, x):
        if len(np.array(x).shape) == 1:
            x = [x]
        pred = np.zeros((len(x)))
        for gc in self.gaussian_components:
            pred += gc(x)
        return pred
    
    
class GMM3(GMM2):
    def predict(self, x):
        if len(np.array(x).shape) == 1:
            x = [x]
        pred = np.zeros((len(x)))
        for gc in self.gaussian_components:
            # pred = np.max(pred, gc(x))
            pred = np.array([max(p,y) for p,y in zip(pred,gc(x))])
        return pred


class CGMM(GMM3):
    def custom_normal(self, x, jpx, Var, jpy, p_thr):
        gn = 1./multivariate_normal.pdf(jpx,jpx,Var)
        pdf = multivariate_normal.pdf(x,jpx,Var)*gn
        # print(pdf)
        # if pdf >= p_thr:
        #     y = (p_thr+(pdf-p_thr)**(1./4))*(1./(p_thr+(1-p_thr)**(1./4)))*jpy
        #     y = max(y,pdf)
        # else:
        #     y = pdf*np.exp(-np.sqrt(p_thr-pdf))*jpy
            
        y = np.where(
            pdf >= p_thr,
            np.maximum(pdf, (p_thr+(pdf-p_thr)**(1./4))*(1./(p_thr+(1-p_thr)**(1./4))))*jpy,
            pdf*np.exp(-np.sqrt(p_thr-pdf))*jpy
        )
        
        return y
    
    def train(self, p_thr=0.7): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        self.extract_jps()
        Var = np.diag(self.diag_sigma)**2
        for jpx, jpy in zip(np.array(self.jumppoints["X"]), np.array(self.jumppoints["Y"])):
            self.gaussian_components.append(lambda x,jpx=jpx,Var=Var,jpy=jpy,p_thr=p_thr: self.custom_normal(x,jpx,Var,jpy,p_thr))

    
class ObservationReward:
    def __init__(self, observations, diag_sigma):
        self.X = observations
        self.diag_sigma = diag_sigma
        self.r_components = []
        
    def setup(self):
        Var = np.diag(self.diag_sigma)**2
        for obs_x in self.X:
            r_component= lambda in_x, obs_x=obs_x: multivariate_normal.pdf(in_x, obs_x, Var)*(1./multivariate_normal.pdf(obs_x, obs_x, Var))*1.0
            self.r_components.append(r_component)
    
    def calc_reward(self, x, penalty = -1):
        if penalty > 0:
            raise(Exception("penalty should be 0 or less."))
        if len(np.array(x).shape) == 1:
            x = list(x)
        r = penalty*np.ones((len(x)))
        for rc in self.r_components:
            r += abs(penalty)*rc(x)
        r = np.minimum(0., r)
        return r
    
    
class UnobservedSD:
    def __init__(self, observations, diag_sigma, penalty = 0.3):
        self.X = observations
        self.diag_sigma = diag_sigma
        self.penalty = penalty
        self.sd_components = []
        if self.penalty <= 0:
            raise(Exception("penalty should be 0 or more. Default value is 0.3."))
        
    def setup(self):
        Var = np.diag(self.diag_sigma)**2
        for obs_x in self.X:
            sd_component= lambda in_x, obs_x=obs_x: -multivariate_normal.pdf(in_x, obs_x, Var)*(1./multivariate_normal.pdf(obs_x, obs_x, Var))*1.0
            self.sd_components.append(sd_component)
    
    def calc_sd(self, x):
        if len(np.array(x).shape) == 1:
            x = [x]
        sd = self.penalty*np.ones((len(x)))
        for sdc in self.sd_components:
            sd += self.penalty*sdc(x)
        sd = np.maximum(0., sd)
        return sd

        

def Run(ct, *args):
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "t0.1/8000/t1" if len(args) == 0 else args[0]
    num_ep = 8000
    n_rand_sample = 50000
    n_learn_step = 1
    max_smsz = 0.8
    nn_options = {
        'n_units': [2] + [200, 200] + [1],
        'n_units_err': [2] + [200, 200] + [1],
        # 'loss_stddev_stop_err': 1.0e-4,
        'error_loss_neg_weight': 0.1,
        # 'loss_stddev_stop': 1.0e-6,
        'num_check_stop': 50,
        "batchsize": 100,
    }
    n_save_ep = num_ep
    
    fixed_input = [
        (),
    ]
    
    use_gmm = False
    gmm_lam = lambda nnmodel: GMM3(nnmodel, diag_sigma=[(1.0-0.1)/33.3, (0.8-0.3)/33.3], Gerr = 1.0)
    gain_pairs = (1.0, 0.5)
    LCB_ratio = 0.0
    
    logdir = base_logdir + "logs/{}/".format(name)
    modeldir = logdir + "{}/".format("models")
    
    if os.path.exists(logdir+"dm.pickle"):
        dm = Domain.load(logdir+"dm.pickle")
        dm.nnmodel.modeldir = modeldir
        dm.nnmodel.nn_options = nn_options
        dm.nnmodel.setup()
    else:
        nnmodel = NNModel(modeldir, nn_options)
        nnmodel.setup()
        # gmm = GMM(nnmodel, diag_sigma=[(1.0-0.1)/50, (0.8-0.3)/50], Gerr = Gerr)
        gmm = gmm_lam(nnmodel)
        dm = Domain(nnmodel, gmm, logdir, use_gmm = use_gmm, LCB_ratio = LCB_ratio, gain_pairs = gain_pairs)
        dm.setup()
    
    while len(dm.log["ep"]) < num_ep:
        dm.execute(n_rand_sample = n_rand_sample, n_learn_step = n_learn_step, max_smsz = max_smsz)
        if len(dm.log["ep"])%n_save_ep==0:
            dm.save()
        