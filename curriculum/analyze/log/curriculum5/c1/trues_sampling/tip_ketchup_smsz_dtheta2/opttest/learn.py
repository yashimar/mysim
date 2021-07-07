#coding: UTF-8
from core_tool import *
from ay_py.core import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from scipy.stats import multivariate_normal
from .......util import *
from ........tasks_domain.util import Rmodel
from ..greedyopt import *


RFUNC = "r_func"


def Help():
    pass


class Domain:
    def __init__(self, nnmodel, logdir, n_rand_sample = 5, n_learn_step = 1):
        self.smsz = np.linspace(0.3,0.8,100)
        self.dtheta2 = np.linspace(0.1,1,100)[::-1]
        self.datotal = {TRUE: np.ones((100,100))*(-100), RFUNC: np.ones((100,100))*(-100)}
        self.nnmodel = nnmodel
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
        }
        self.logdir = logdir
        self.n_rand_sample = n_rand_sample
        self.n_learn_step = n_learn_step

    def setup(self):
        self.datotal[TRUE] = np.load(SRC_PATH+"datotal.npy")
        for i in range(100):
            for j in range(100):
                self.datotal[RFUNC][i,j] = rfunc(self.datotal[TRUE][i,j].item())
        
    def execute(self):
        ep = len(self.nnmodel.model.DataX)
        print("ep: {}".format(ep))
        
        idx_smsz = RandI(len(self.smsz))
        smsz = self.smsz[idx_smsz]
        true_opt_dtheta2_idx = np.argmax(self.datotal[RFUNC][:, idx_smsz])
        true_opt_dtheta2 = self.dtheta2[true_opt_dtheta2_idx]
        true_opt_r = self.datotal[RFUNC][true_opt_dtheta2_idx, idx_smsz]
        
        if ep <= self.n_rand_sample:
            idx_est_opt_dtheta2 = RandI(len(self.dtheta2))
            est_opt_dtheta2 = self.dtheta2[idx_est_opt_dtheta2]
            est_datotal = 0
            est_opt_Er = 0
        else:
            est_nn_Er, est_nn_Sr = [], []
            for dtheta2 in self.dtheta2:
                est_datotal = self.nnmodel.model.Predict(x=[dtheta2, smsz], with_var=True)
                est_nn = Rmodel("Fdatotal_gentle").Predict(x=[0.3, est_datotal.Y[0].item()], x_var=[0, est_datotal.Var[0].item()], with_var=True)
                est_nn_Er.append(est_nn.Y.item())
                est_nn_Sr.append(np.sqrt(est_nn.Var[0,0]).item())
            idx_est_opt_dtheta2 = np.argmax(est_nn_Er)
            est_opt_dtheta2 = self.dtheta2[idx_est_opt_dtheta2]
            est_datotal = self.nnmodel.model.Predict(x=[est_opt_dtheta2, smsz], with_var=True).Y[0].item()
            est_opt_Er = est_nn_Er[idx_est_opt_dtheta2]
        
        true_datotal = self.datotal[TRUE][idx_est_opt_dtheta2, idx_smsz]
        true_r_at_est_opt_dthtea2 = rfunc(true_datotal)
        
        if ep % self.n_learn_step == 0:
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
        
        with open(self.logdir+"log.yaml", "w") as f:
            yaml.dump(self.log, f)
            
    @classmethod
    def load(self, path):
        with open(path, mode="rb") as f:
            dm = pickle.load(f)
        return dm
    
    @classmethod
    def save(self, dm,path):
        with open(path, mode="wb") as f:
            pickle.dump(dm, f)
        
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
    def __init__(self, nnmodel):
        self.nnmodel = nnmodel
        self.jumppoints = {"X": [], "Y": []}
        self.gaussian_components = []
    
    def extract_jps(self, Gerr):
        model = self.nnmodel.model
        for i, (x, y) in enumerate(zip(model.DataX, model.DataY)):
            p_mean = model.Forward(x_data = model.DataX[i:i+1], train = False).data.item() #Chainerのバグ回避用
            p_err = model.ForwardErr(x_data = model.DataX[i:i+1], train = False).data.item()
            if (y < (p_mean - Gerr*p_err) or (y > (p_mean + Gerr*p_err))):
                self.jumppoints["X"].append(x)
                self.jumppoints["Y"].append(abs(y - p_mean))
                
    def train(self, diag_sigma, Gerr = 1.0): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        from copy import deepcopy
        self.extract_jps(Gerr)
        tmp = []
        Var = np.diag(diag_sigma)**2
        for jpx, jpy in zip(self.jumppoints["X"], self.jumppoints["Y"]):
            self.gaussian_components.append(lambda x,jpx=jpx,jpy=jpy: multivariate_normal.pdf(x,jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy)
    
    def predict(self, x):
        pred = 0
        for gc in self.gaussian_components:
            pred += gc(x)
        return pred
            

def Run(ct, *args):
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "Er/t0.1_fixed"
    num_ep = 500
    nn_options = {
        'n_units': [2] + [200, 200] + [1],
        'n_units_err': [2] + [200, 200] + [1],
        # 'loss_stddev_stop_err': 1.0e-4,
        'error_loss_neg_weight': 0.1,
    }
    
    logdir = base_logdir + "logs/{}/".format(name)
    modeldir = logdir + "{}/".format("models")
    
    if os.path.exists(logdir+"dm.pickle"):
        dm = Domain.load(logdir+"dm.pickle")
    else:
        nnmodel = NNModel(modeldir, nn_options)
        nnmodel.setup()
        dm = Domain(nnmodel, logdir, n_rand_sample = 50000, n_learn_step = 1)
        dm.setup()
    
    for i in range(num_ep):
        dm.execute()
        Domain.save(dm, logdir+"dm.pickle")
        