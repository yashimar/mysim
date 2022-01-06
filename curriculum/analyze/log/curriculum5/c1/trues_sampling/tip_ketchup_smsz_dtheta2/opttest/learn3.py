#coding: UTF-8
from learn2 import *
from .setup2 import *
import cma
from scipy.optimize import fmin_l_bfgs_b
from scipy import optimize
from scipy.special import gammaln
import shutil
from math import pi, sqrt, gamma
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import gaussian_kde
from scipy.spatial import distance
from scipy import interpolate


TIP = "tip"
SHAKE = "shake"


class GMM7(GMM5, object):
    def train(self, recreate_jp = True): #引数diag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
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


class GMM8(GMM7, object):
    def __init__(self, nnmodel, diag_sigma, options, lam = 0.0, Gerr = 1.0, w_positive = False):
        super(GMM8, self).__init__(nnmodel, diag_sigma, Gerr)
        self.options = options
        self.w_positive = w_positive
        
    def extract_jps(self):
        self.jumppoints = {"X": [], "Y": []}
        model = self.nnmodel.model
        uq_DataX, uq_idx = np.unique(model.DataX, axis = 0, return_index = True)
        p_means = model.Forward(x_data = uq_DataX, train = False).data
        p_errs = model.ForwardErr(x_data = uq_DataX, train = False).data
        for i, (p_mean, p_err, x, y) in enumerate(zip(p_means, p_errs, uq_DataX, model.DataY[uq_idx])):
            if (y < (p_mean - self.Gerr*p_err) or (y > (p_mean + self.Gerr*p_err))):
                jp = max((p_mean - self.Gerr*p_err)-y, y-(p_mean + self.Gerr*p_err))
                self.jumppoints["X"].append(x.tolist())
                self.jumppoints["Y"].append(jp.tolist())
    
    def opt_funciton(self, w, X, y, tau, lam):
        if self.w_positive:
            w = np.maximum(0,w) #wの定義域を正にするため
            # w = np.exp(w)
        delta = y - X.dot(w)
        indic = np.array(delta <= 0., dtype=np.float32)
        rho = (tau - indic)*delta
        evaluation = sum(rho) + lam * w.T.dot(w)
        # if self.w_positive:
        #     evaluation += (-100)*np.minimum(0,w)
        return evaluation
    
    def train(self, recreate_jp = True): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        tau = self.options['tau']
        lam = self.options['lam']
        maxiter = self.options['maxiter']
        verbose = self.options['verbose']
        popsize = self.options['popsize']
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
            return 0
        elif len(X) == 1:
            self.w_concat = [1]
            return 0
        else:
            # if self.w_positive:
            #     mean_init = [0.8] * len(y)
            #     # mean_init = np.where(y<0.04, 0, 0.7)
            #     sd_init = 0.3
            #     # mean_init = [0.5] * len(y)
            #     # sd_init = 0.3
            # else:
            #     mean_init = [0.] * len(y)
            #     sd_init = 0.5
            mean_init, _ = nnls(X, y)
            sd_init = 0.3
            es = cma.CMAEvolutionStrategy(mean_init, sd_init, {'seed':0, 'maxiter': maxiter, 'popsize': popsize, 'tolfacupx': 1e8})
            f = lambda w: self.opt_funciton(w, X, y, tau, lam)
            es.optimize(f, verbose = verbose)
            self.w_concat = es.result[0]
            if self.w_positive:
                self.w_concat = np.maximum(0,self.w_concat)
            return es.result[1]
        
    def predict(self, x):
        if len(np.array(x).shape) == 1:
            x = [x]
        pred = np.zeros((len(x)))
        for gc, w in zip(self.gc_concat, self.w_concat):
            pred += w*gc(x)
        # pred = np.maximum(0, pred)
        return pred


class GMM9(GMM8, object):
    def opt_function(self, w, X, y, tau, lam):
        # y = y.T
        # w = w.T
        # print(y.shape, w.shape, X.shape)
        # w = np.maximum(0, w)
        # tau = 0.9
        # lam = 1e-6
        delta = y - X.dot(w)
        indic = np.where(delta<=0, 1, 0)
        rho = (tau - indic)*delta
        
        f = sum(rho) + lam*w.T.dot(w)
        # fprime = (-1./len(y)*X.T.dot(np.abs(tau - indic)*np.sign(delta)) + 2*lam*w)*np.where(w>0, 1, 0)
        fprime = -1./len(y)*(np.abs(tau - indic)*np.sign(delta)).dot(X) + 2*lam*w
        # f = delta.T.dot(delta) + lam*w.T.dot(w)
        # fprime = - X.T.dot(y) + X.T.dot(X).dot(w)
        
        return f, fprime
    
    def train(self, recreate_jp = True, init_w = np.ones): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        tau = self.options['tau']
        lam = self.options['lam']
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
            return 0
        else:
            f = lambda w: self.opt_function(w, X, y, tau, lam)[0]
            fprime = lambda w: self.opt_function(w, X, y, tau, lam)[1]
            self.w_concat, fmin, d = fmin_l_bfgs_b(f,
                                                   init_w(len(y)),
                                                #    np.ones(len(y)), 
                                                    # np.zeros(len(y)),
                                                #    fprime = fprime,   #何故か当てはまりが数値微分より悪いので放置
                                                    pgtol = 1e-8,
                                                    # bounds = [(0, None)]*len(y),
                                                    approx_grad = 1,
                                                    maxiter = 1e5,
                                                    maxfun = 1e5, 
                                                    # maxls = 10000
                                                   )
            return fmin
            
    def predict(self, x):
        if len(np.array(x).shape) == 1:
            x = [x]
        pred = np.zeros((len(x)))
        for gc, w in zip(self.gc_concat, self.w_concat):
            pred += w*gc(x)
        pred = np.maximum(0, pred)
        return pred
    
    
class GMM9v2(GMM9, object):
    def opt_function(self, w, X, y, tau, lam):
        # y = y.T
        # w = w.T
        # print(y.shape, w.shape, X.shape)
        # w = np.maximum(0, w)
        # tau = 0.9
        # lam = 1e-6
        w = np.where(w<0,0,w)
        delta = y - X.dot(w)
        indic = np.where(delta<=0, 1, 0)
        rho = (tau - indic)*delta
        
        f = sum(rho) + lam*w.T.dot(w)
        # fprime = (-1./len(y)*X.T.dot(np.abs(tau - indic)*np.sign(delta)) + 2*lam*w)*np.where(w>0, 1, 0)
        fprime = -1./len(y)*(np.abs(tau - indic)*np.sign(delta)).dot(X) + 2*lam*w
        # f = delta.T.dot(delta) + lam*w.T.dot(w)
        # fprime = - X.T.dot(y) + X.T.dot(X).dot(w)
        
        return f, fprime
    
    def train(self, recreate_jp = True, init_w = np.ones): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        tau = self.options['tau']
        lam = self.options['lam']
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
            return 0
        else:
            f = lambda w: self.opt_function(w, X, y, tau, lam)[0]
            fprime = lambda w: self.opt_function(w, X, y, tau, lam)[1]
            self.w_concat, fmin, d = fmin_l_bfgs_b(f,
                                                #    init_w(len(y)),
                                                #    np.ones(len(y)), 
                                                    np.zeros(len(y)),
                                                #    fprime = fprime,   #何故か当てはまりが数値微分より悪いので放置
                                                    pgtol = 1e-8,
                                                    # bounds = [(0, None)]*len(y),
                                                    approx_grad = 1,
                                                    maxiter = 1e5,
                                                    maxfun = 1e5, 
                                                    # maxls = 10000
                                                   )
            self.w_concat = np.where(self.w_concat<0,0,self.w_concat)
            return fmin
    
    
class TMM(GMM9, object):
    def multivariate_t(self, x, mean, shape, df):
        dim = mean.size

        vals, vecs = np.linalg.eigh(shape)
        logdet     = np.log(vals).sum()
        valsinv    = np.array([1./v for v in vals])
        U          = vecs * np.sqrt(valsinv)
        dev        = x - mean
        maha       = np.square(np.dot(dev, U)).sum(axis=-1)

        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim/2. * np.log(df * np.pi)
        D = 0.5 * logdet
        E = -t * np.log(1 + (1./df) * maha)

        return np.exp(A - B - C - D + E)
     
    def train(self, recreate_jp = True): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        tau = self.options['tau']
        lam = self.options['lam']
        if recreate_jp:
            self.extract_jps()
        self.gc_concat = []
        self.w_concat = []
        Var = np.diag(self.diag_sigma)**2
        for jpx, jpy in zip(np.array(self.jumppoints["X"]), np.array(self.jumppoints["Y"])):
            self.gc_concat.append(
                lambda x,jpx=jpx,jpy=jpy: self.multivariate_t(x,jpx,Var,1)*(1./self.multivariate_t(np.array([jpx]),jpx,Var,1))*jpy
            )
        y = np.array(self.jumppoints["Y"]).flatten()
        X = np.array([[gc(x).item() for gc in self.gc_concat] for x in self.jumppoints["X"]])
        if len(X) == 0:
            self.w_concat = []
        else:
            f = lambda w: self.opt_function(w, X, y, tau, lam)[0]
            fprime = lambda w: self.opt_function(w, X, y, tau, lam)[1]
            self.w_concat, fmin, d = fmin_l_bfgs_b(f, np.ones(len(y)), 
                                                #    fprime = fprime,   #何故か当てはまりが数値微分より悪いので放置
                                                    pgtol = 1e-8,
                                                    # bounds = [(0, None)]*len(y),
                                                    approx_grad = 1,
                                                    maxiter = 1e5,
                                                    maxfun = 1e5, 
                                                    # maxls = 10000
                                                   )


class GMM10(GMM9, object):
    def train(self, recreate_jp = True): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        tau = self.options['tau']
        maxiter = self.options['maxiter']
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
            return 0
        else:
            model = QuantReg(y, X)
            results = model.fit(q = tau, max_iter = maxiter)
            # results = model.fit_regularized(q = tau, max_iter = maxiter)
            self.w_concat = results.params
            return self.opt_function(self.w_concat, X, y, tau, 0)[0]


class GMM11(GMM8, object):
    def opt_funciton(self, w, X, y, tau, lam):
        if self.w_positive:
            w = np.maximum(0,w) #wの定義域を正にするため
            # w = np.exp(w)
        delta = y - X.dot(w)
        indic = np.array(delta <= 0., dtype=np.float32)
        rho = (tau - indic)*delta
        evaluation = sum(rho) + lam * w.T.dot(w)
        # if self.w_positive:
        #     evaluation += (-100)*np.minimum(0,w)
        return evaluation
    
    def train(self, recreate_jp = True): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        tau = self.options['tau']
        lam = self.options['lam']
        maxiter = self.options['maxiter']
        verbose = self.options['verbose']
        popsize = self.options['popsize']
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
            return 0
        elif len(X) == 1:
            self.w_concat = [1]
            return 0
        else:
            tmp_gmm = GMM9v2(None, diag_sigma=self.diag_sigma, options = {"tau": tau, "lam": 0})
            tmp_gmm.jumppoints.update(self.jumppoints)
            tmp_gmm.train(recreate_jp = False)
            
            mean_init = tmp_gmm.w_concat
            sd_init = 0.3
            es = cma.CMAEvolutionStrategy(mean_init, sd_init, {'seed':0, 'maxiter': maxiter, 'popsize': popsize, 'tolfacupx': 1e8})
            f = lambda w: self.opt_funciton(w, X, y, tau, lam)
            es.optimize(f, verbose = verbose)
            self.w_concat = es.result[0]
            if self.w_positive:
                self.w_concat = np.maximum(0,self.w_concat)
            return es.result[1]
        
    def predict(self, x):
        if len(np.array(x).shape) == 1:
            x = [x]
        pred = np.zeros((len(x)))
        for gc, w in zip(self.gc_concat, self.w_concat):
            pred += w*gc(x)
        # pred = np.maximum(0, pred)
        return pred


class GMM12(GMM8, object):
    def opt_funciton(self, w, X, y, tau, lam):
        delta = y - X.dot(w)
        indic = np.array(delta <= 0., dtype=np.float32)
        rho = (tau - indic)*delta
        evaluation = sum(rho) + lam * w.T.dot(w)
        return evaluation
    
    def train(self, recreate_jp = True): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        tau = self.options['tau']
        lam = self.options['lam']
        maxiter = self.options['maxiter']
        verbose = self.options['verbose']
        popsize = self.options['popsize']
        tol = self.options['tol']
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
            return 0
        elif len(X) == 1:
            self.w_concat = [1]
            return 0
        else:
            f = lambda w: self.opt_funciton(w, X, y, tau, lam)
            if self.w_positive:
                bounds = [[0.0, 2.0] for _ in range(len(X))]
            else:
                bounds = [[-2.0, 2.0] for _ in range(len(X))]
            res = optimize.differential_evolution(f, bounds, strategy='best1bin', maxiter=int(maxiter), popsize=popsize, tol=tol)
            self.w_concat = res.x
            
            return res.fun.item()

            
def gmm_test():
    pref = "onpolicy2/GMM9Sig8LCB4/checkpoints"
    logdir = BASE_DIR + "opttest/logs/{}/t84/ch500/".format(pref)
    dm = Domain2.load(logdir+"dm.pickle")
    
    p = 8
    # lam = 1e-5
    Var = np.diag([(0.8-0.3)/(100./p)])**2
    # gmm = GMM5(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], lam = 1e-4)
    # gmm = GMM4(dm.nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)])
    # gmm = GMM7(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)])
    # gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./5), (0.8-0.3)/(100./5)], options = {"tau": 0.9, "lam": 1e-6})
    # gmm = TMM(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./5), (0.8-0.3)/(100./5)], options = {"tau": 0.9, "lam": 1e-6})
    # gmm = GMM9(dm.nnmodels[SHAKE], diag_sigma=[(0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 0})
    # gmm = GMM9(dm.nnmodels[SHAKE], diag_sigma=[(0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-2})
    # gmm = GMM8(dm.nnmodels[SHAKE], diag_sigma=[(0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 0, 'maxiter': 1e2, 'verbose': 0})
    gmm = GMM10(dm.nnmodels[SHAKE], diag_sigma=[(0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 0, 'maxiter': 1e2})
    # gmm.extract_jps()
    # gmm.jumppoints.update({"X": [[0., x[1]] for x in gmm.jumppoints["X"]]})
    # tx = np.array([x[1] for x in gmm.jumppoints["X"]])
    # uq = np.unique(tx, return_index=True)[1][[0,1,2,7,8,9,10,11,12]]
    # # uq = np.unique(tx, return_index=True)[1]
    # tx = np.array([x[1] for x in gmm.jumppoints["X"]])
    # gmm.jumppoints.update({"X": [x for i, x in enumerate(gmm.jumppoints["X"]) if i in uq]})
    # gmm.jumppoints.update({"Y": [y for i, y in enumerate(gmm.jumppoints["Y"]) if i in uq]})
    # gmm.train(recreate_jp = False)
    gmm.train()
    # min_loss = gmm.train()
    # min_loss = gmm.train(init_w = np.zeros)
    # print(min_loss)
    
    x_list = np.linspace(0.3, 0.8, 1000)
    # X = np.array([[0., x] for x in x_list])
    X = np.array([[x] for x in x_list])
    P = gmm.predict(X)
    
    ttx = np.array(gmm.jumppoints["X"])
    tty = np.array(gmm.jumppoints["Y"])
    
    jpmeta = [[(multivariate_normal.pdf([x],jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy).item() for x in x_list] for jpx, jpy in zip(ttx, tty)]
    # jpx = ttx[0]
    # jpy = tty[0]
    # for x in x_list:
    #     tmp = multivariate_normal.pdf([0., x],jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy
    #     tmp = tmp.item()
    #     print(x, tmp)
    # # print(jpmeta[0])
    # print(jpx)
    # jpge
    
    # tttx = [x[1] for x in ttx]
    tttx = [x[0] for x in ttx]
    fig = plt.figure()
    for jp in jpmeta:
        plt.scatter(x_list, jp, c="skyblue")
    plt.scatter(x_list, P, c="purple")
    plt.scatter(tttx, tty, c="orange")
    # plt.scatter(tttx, gmm.predict(ttx))
    plt.xlim(0.3,0.8)
    plt.show()
    
    fig = plt.figure()
    plt.bar(left = tttx, height = gmm.w_concat, color="red", width = 0.01)
    plt.xlim(0.3,0.8)
    plt.show()


def gmm_test2():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    name = "GMM9Sig5LCB3/checkpoints/t1/ch500"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/{}/".format(name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    datotal = setup_datotal(dm, logdir)
    # gmm = GMM7(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)])
    # gmm = GMM8(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./5), (0.8-0.3)/(100./5)], options = {"tau": 0.9, "lam": 1e-4, "maxiter": 1e3, "verbose": 1})
    # gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./5), (0.8-0.3)/(100./5)], options = {"tau": 0.9, "lam": 1e-6})
    # gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-6})
    gmm = TMM(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-6})
    gmm.train()
    X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ]).astype(np.float32)
    gmmpred = gmm.predict(X).reshape(100,100)
    
    diffcs = [
        [0, "rgb(0, 0, 0)"],
        [0.01, "rgb(255, 255, 200)"],
        [1, "rgb(255, 0, 0)"],
    ]
    jpx_idx = [[idx_of_the_nearest(dm.dtheta2, x[0]), idx_of_the_nearest(dm.smsz, x[1])] for x in np.array(gmm.jumppoints["X"])]
    jpx_tr = [dm.datotal[TIP][RFUNC][idx[0],idx[1]] for idx in jpx_idx]
    jpx_gmm = [gmmpred[idx[0],idx[1]] for idx in jpx_idx]
    jpy = [y[0] for y in gmm.jumppoints["Y"]]
    linex = [[x,x] for x in np.array(gmm.jumppoints["X"])[:,1]]
    liney = [[y,y] for y in np.array(gmm.jumppoints["X"])[:,0]]
    linegmm =[[a,b] for a, b in zip(jpy, jpx_gmm)]
        
    fig = go.Figure()
    fig.add_trace(go.Surface(
        z = gmmpred, x = dm.smsz, y = dm.dtheta2,
        cmin = 0, cmax = 0.2, colorscale = diffcs,
        showlegend = False,
    ))
    fig.add_trace(go.Scatter3d(
        z = jpy, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
        marker = dict(
            color = "red",
            size = 4,
        )
    ))
    for tz,tx,ty in zip(linegmm, linex, liney):
        fig.add_trace(go.Scatter3d(
            z = tz, x = tx, y = ty,
            mode = "lines",
            line = dict(
                color = "red",
            ),
            showlegend = False,
        ))
    fig['layout']['scene']['xaxis']['title'] = "size_srcmouth" 
    fig['layout']['scene']['yaxis']['title'] = "dtheta2" 
    fig['layout']['scene']['zaxis']['title'] = "gmm predict" 
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "curve_tip.html", auto_open=False)
    
    
def gmm_test3():
    n_dummys = 50
    njp_range = (5, 11)
    jpx_range = (0.3, 0.8)
    jpy_range = (0.01, 0.1/3)
    Var = np.diag([(0.8-0.3)/(100./8)])**2
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/"
    
    dummys = {"jpx": [], "jpy": [], "func": []}
    for i in range(n_dummys):
        njp = np.random.randint(*njp_range)
        jpx, jpy = [], []
        for _ in range(njp):
            jpx.append(np.random.uniform(*jpx_range))
            while True:
                tmp = np.random.normal(*jpy_range)
                if tmp >= jpy_range[0]:
                    jpy.append(tmp)
                    break
        func = lambda x, jpx=jpx, jpy=jpy: np.max([multivariate_normal.pdf([x],jpx_i,Var)*(1./multivariate_normal.pdf(jpx_i,jpx_i,Var))*jpy_i for jpx_i,jpy_i in zip(jpx, jpy)])
        dummys["jpx"].append(jpx)
        dummys["jpy"].append(jpy)
        dummys["func"].append(func)
    
        xlist = np.linspace(0.3,0.8,100)
        ylist = [func(x) for x in xlist]
        check_or_create_dir(save_img_dir+"test")
        fig = plt.figure()
        plt.scatter(jpx, jpy, color="orange")
        plt.plot(xlist, ylist, color="skyblue")
        plt.savefig(save_img_dir+"test/t{}.png".format(i))
        plt.close()
    
    savedir = BASE_DIR + "opttest/gmm_test/"
    check_or_create_dir(savedir)
    with open(savedir+"dummys.pickle", mode="wb") as f:
        dill.dump(dummys, f)
        

def gmm_test4():
    n_dummys = 50
    Var = np.diag([(0.8-0.3)/(100./8)])**2
    
    with open(BASE_DIR + "opttest/gmm_test/dummys.pickle", "rb") as f:
        dummys = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/GMM9_init1/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    for i in range(n_dummys):
        jpx = dummys["jpx"][i]
        jpy = dummys["jpy"][i]
        func = dummys["func"][i]
        
        gmm = GMM9(None, diag_sigma=np.sqrt(Var), options = {"tau": 0.9, "lam": 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        min_loss = gmm.train(recreate_jp = False)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        # fig = plt.figure()
        # plt.scatter(jpx, jpy, color="orange")
        # plt.plot(xlist, ylist, color="skyblue")
        # plt.plot(xlist, plist, color="purple")
        # plt.savefig(save_img_dir+"t{}.png".format(i))
        # plt.close()
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)
    
    
def gmm_test5():
    n_dummys = 50
    Var = np.diag([(0.8-0.3)/(100./8)])**2
    
    with open(BASE_DIR + "opttest/gmm_test/dummys.pickle", "rb") as f:
        dummys = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/GMM9_init0/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    for i in range(n_dummys):
        jpx = dummys["jpx"][i]
        jpy = dummys["jpy"][i]
        func = dummys["func"][i]
        
        gmm = GMM9(None, diag_sigma=np.sqrt(Var), options = {"tau": 0.9, "lam": 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        min_loss = gmm.train(recreate_jp = False, init_w = np.zeros)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        # fig = plt.figure()
        # plt.scatter(jpx, jpy, color="orange")
        # plt.plot(xlist, ylist, color="skyblue")
        # plt.plot(xlist, plist, color="purple")
        # plt.savefig(save_img_dir+"t{}.png".format(i))
        # plt.close()
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)
    
    
def gmm_test5v2():
    n_dummys = 50
    Var = np.diag([(0.8-0.3)/(100./8)])**2
    
    with open(BASE_DIR + "opttest/gmm_test/dummys.pickle", "rb") as f:
        dummys = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/GMM9_init_by_cmaes/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    for i in [10]:
        jpx = dummys["jpx"][i]
        jpy = dummys["jpy"][i]
        func = dummys["func"][i]
        
        gmm = GMM8(None, diag_sigma=np.sqrt(Var), options = {"tau": 0.9, "lam": 0, 'maxiter': 1e2, 'verbose': 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        gmm.train(recreate_jp = False)
        init_w = lambda _, gmm=gmm: gmm.w_concat
        
        gmm = GMM9(None, diag_sigma=np.sqrt(Var), options = {"tau": 0.9, "lam": 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        min_loss = gmm.train(recreate_jp = False, init_w = init_w)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        fig = plt.figure()
        plt.scatter(jpx, jpy, color="orange")
        plt.plot(xlist, ylist, color="skyblue")
        plt.plot(xlist, plist, color="purple")
        plt.savefig(save_img_dir+"t{}.png".format(i))
        plt.close()
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)
    
    
def gmm_test6():
    n_dummys = 50
    Var = np.diag([(0.8-0.3)/(100./8)])**2
    
    with open(BASE_DIR + "opttest/gmm_test/dummys.pickle", "rb") as f:
        dummys = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/GMM8_init0/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    for i in range(n_dummys):
    # for i in [6,9]:
        jpx = dummys["jpx"][i]
        jpy = dummys["jpy"][i]
        func = dummys["func"][i]
        
        gmm = GMM8(None, diag_sigma=np.sqrt(Var), options = {"tau": 0.9, "lam": 0, 'maxiter': 2e2, 'popsize': 100, 'verbose': 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        min_loss = gmm.train(recreate_jp = False)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        fig = plt.figure()
        plt.scatter(jpx, jpy, color="orange")
        plt.plot(xlist, ylist, color="skyblue")
        plt.plot(xlist, plist, color="purple")
        plt.savefig(save_img_dir+"t{}.png".format(i))
        plt.close()
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)
    
    
def gmm_test7():
    n_dummys = 50
    Var = np.diag([(0.8-0.3)/(100./8)])**2
    
    with open(BASE_DIR + "opttest/gmm_test/dummys.pickle", "rb") as f:
        dummys = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/GMM10/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    for i in range(n_dummys):
        jpx = dummys["jpx"][i]
        jpy = dummys["jpy"][i]
        func = dummys["func"][i]
        
        gmm = GMM10(None, diag_sigma=np.sqrt(Var), options = {"tau": 0.9, 'maxiter': 1e2})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        min_loss = gmm.train(recreate_jp = False)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        fig = plt.figure()
        plt.scatter(jpx, jpy, color="orange")
        plt.plot(xlist, ylist, color="skyblue")
        plt.plot(xlist, plist, color="purple")
        plt.savefig(save_img_dir+"t{}.png".format(i))
        plt.close()
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)


def gmm_test8():
    n_dummys = 50
    Var = np.diag([(0.8-0.3)/(100./8)])**2
    
    with open(BASE_DIR + "opttest/gmm_test/dummys.pickle", "rb") as f:
        dummys = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/GMM8_ridge/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    for i in range(n_dummys):
    # for i in [6,9]:
        jpx = dummys["jpx"][i]
        jpy = dummys["jpy"][i]
        func = dummys["func"][i]
        
        gmm = GMM8(None, diag_sigma=np.sqrt(Var), options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 2e2, 'popsize': 100, 'verbose': 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        min_loss = gmm.train(recreate_jp = False)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        fig = plt.figure()
        plt.scatter(jpx, jpy, color="orange")
        plt.plot(xlist, ylist, color="skyblue")
        plt.plot(xlist, plist, color="purple")
        plt.savefig(save_img_dir+"t{}.png".format(i))
        plt.close()
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)
    
    
def gmm_test9():
    n_dummys = 50
    Var = np.diag([(0.8-0.3)/(100./8)])**2
    
    with open(BASE_DIR + "opttest/gmm_test/dummys.pickle", "rb") as f:
        dummys = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/GMM8_positive/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    for i in range(n_dummys):
    # for i in [38]:
        jpx = dummys["jpx"][i]
        jpy = dummys["jpy"][i]
        func = dummys["func"][i]
        
        gmm = GMM8(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 0, 'maxiter': 2e3, 'popsize': 100, 'verbose': 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        min_loss = gmm.train(recreate_jp = False)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        fig = plt.figure()
        plt.scatter(jpx, jpy, color="orange")
        plt.plot(xlist, ylist, color="skyblue")
        plt.plot(xlist, plist, color="purple")
        plt.savefig(save_img_dir+"t{}.png".format(i))
        plt.close()
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)
    
    
def gmm_test10():
    n_dummys = 50
    Var = np.diag([(0.8-0.3)/(100./8)])**2
    
    with open(BASE_DIR + "opttest/gmm_test/dummys.pickle", "rb") as f:
        dummys = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/GMM8_positive_ridge/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    for i in range(n_dummys):
    # for i in [38]:
        jpx = dummys["jpx"][i]
        jpy = dummys["jpy"][i]
        func = dummys["func"][i]
        
        gmm = GMM8(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-2, 'maxiter': 2e3, 'popsize': 100, 'verbose': 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        min_loss = gmm.train(recreate_jp = False)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        fig = plt.figure()
        plt.scatter(jpx, jpy, color="orange")
        plt.plot(xlist, ylist, color="skyblue")
        plt.plot(xlist, plist, color="purple")
        plt.savefig(save_img_dir+"t{}.png".format(i))
        plt.close()
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)
    
    
def gmm_test11():
    n_samples = 100
    pref_list = [
        (lambda i: BASE_DIR + "opttest/logs/onpolicy2/GMM9Sig5LCB3/checkpoints/t{}/ch500/".format(i), 5),
        (lambda i: BASE_DIR + "opttest/logs/onpolicy2/GMM9Sig8LCB4/checkpoints/t{}/ch500/".format(i), 8),
    ]
    
    samples = {"jpx": [], "jpy": [], "func": [], "logdir": [], "p": []}
    for _ in range(n_samples):
        i = np.random.randint(1,99)
        pref, p = pref_list[RandI(len(pref_list))]
        logdir = pref(i)
        print(_, logdir)
        with open(logdir+"log.yaml", "r") as yml:
            log = yaml.load(yml)
        j = np.random.randint(50,501)
        jpx = np.array(log["est_gmm_JP_shake"][j]["X"])
        jpy = np.array(log["est_gmm_JP_shake"][j]["Y"])
        Var = np.diag([(0.8-0.3)/(100./p)])**2
        func = lambda x, jpx=jpx, jpy=jpy, Var=Var: np.max([multivariate_normal.pdf(x,jpx_i,Var)*(1./multivariate_normal.pdf(jpx_i,jpx_i,Var))*jpy_i for jpx_i,jpy_i in zip(jpx, jpy)])
        samples["jpx"].append(jpx)
        samples["jpy"].append(jpy)
        samples["func"].append(func)
        samples["p"].append(p)
        samples["logdir"].append(logdir+"_"+str(j))

    savedir = BASE_DIR + "opttest/gmm_test/"
    check_or_create_dir(savedir)
    with open(savedir+"shake.pickle", mode="wb") as f:
        dill.dump(samples, f)
        

def gmm_test12():
    n_samples = 100
    
    with open(BASE_DIR + "opttest/gmm_test/shake.pickle", "rb") as f:
        samples = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/shake/GMM8_positive_ridge/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    time_concat = []
    for i in range(n_samples):
    # for i in range(90,100):
        if len(samples["jpx"][i]) == 0:
            train_loss_concat.append(0)
            val_loss_concat.append(0)
            continue
        jpx, uq_index = np.unique(samples["jpx"][i], return_index = True)
        jpy = samples["jpy"][i][uq_index]
        func = samples["func"][i]
        p = samples["p"][i]
        
        Var = np.diag([(0.8-0.3)/(100./p)])**2
        gmm = GMM8(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 100, 'verbose': 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        t1 = time.time()
        min_loss = gmm.train(recreate_jp = False)
        t2 = time.time() - t1
        print(i, "Time: ", t2)
        print(i, "Loss: ", min_loss)
        time_concat.append(t2)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        # fig = plt.figure()
        # plt.scatter(jpx, jpy, color="orange")
        # plt.plot(xlist, ylist, color="skyblue")
        # plt.plot(xlist, plist, color="purple")
        # plt.savefig(save_img_dir+"t{}.png".format(i))
        # plt.close()
        
    print("Time mean: ", np.mean(time_concat))
    print("Time sd: ", np.std(time_concat))
    print("Train Loss mean: ", np.mean(train_loss_concat))
    print("Train Loss sd: ", np.std(train_loss_concat))
    print("Validation Loss mean: ", np.mean(val_loss_concat))
    print("Validation Loss sd: ", np.std(val_loss_concat))
        
    # sort_idx = np.argsort(train_loss_concat)[::-1]
    # x = [str(idx) for idx in sort_idx]
    # y = np.array(train_loss_concat)[sort_idx]
    # fig = go.Figure()
    # fig.add_trace(go.Bar(
    #     x = x, y = y,
    # ))
    # fig['layout']['xaxis']['title'] = "data index"
    # fig['layout']['yaxis']['title'] = "最適化された評価値"
    # plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    # sort_idx = np.argsort(val_loss_concat)[::-1]
    # x = [str(idx) for idx in sort_idx]
    # y = np.array(val_loss_concat)[sort_idx]
    # fig = go.Figure()
    # fig.add_trace(go.Bar(
    #     x = x, y = y,
    # ))
    # fig['layout']['xaxis']['title'] = "data index"
    # fig['layout']['yaxis']['title'] = "validation error"
    # plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)
    
    
def gmm_test12v2():
    n_samples = 100
    
    with open(BASE_DIR + "opttest/gmm_test/shake.pickle", "rb") as f:
        samples = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/shake/GMM12_positive_ridge/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    time_concat = []
    for i in range(n_samples):
    # for i in range(90,100):
        if len(samples["jpx"][i]) == 0:
            train_loss_concat.append(0)
            val_loss_concat.append(0)
            continue
        jpx, uq_index = np.unique(samples["jpx"][i], return_index = True)
        jpy = samples["jpy"][i][uq_index]
        func = samples["func"][i]
        p = samples["p"][i]
        
        Var = np.diag([(0.8-0.3)/(100./p)])**2
        # gmm = GMM8(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 100, 'verbose': 0})
        gmm = GMM12(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        t1 = time.time()
        min_loss = gmm.train(recreate_jp = False)
        t2 = time.time() - t1
        print(i, "Time: ", t2)
        print(i, "Loss: ", min_loss)
        time_concat.append(t2)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        # fig = plt.figure()
        # plt.scatter(jpx, jpy, color="orange")
        # plt.plot(xlist, ylist, color="skyblue")
        # plt.plot(xlist, plist, color="purple")
        # plt.savefig(save_img_dir+"t{}.png".format(i))
        # plt.close()
        
    print("Time mean: ", np.mean(time_concat))
    print("Time sd: ", np.std(time_concat))
    print("Train Loss mean: ", np.mean(train_loss_concat))
    print("Train Loss sd: ", np.std(train_loss_concat))
    print("Validation Loss mean: ", np.mean(val_loss_concat))
    print("Validation Loss sd: ", np.std(val_loss_concat))
        
    # sort_idx = np.argsort(train_loss_concat)[::-1]
    # x = [str(idx) for idx in sort_idx]
    # y = np.array(train_loss_concat)[sort_idx]
    # fig = go.Figure()
    # fig.add_trace(go.Bar(
    #     x = x, y = y,
    # ))
    # fig['layout']['xaxis']['title'] = "data index"
    # fig['layout']['yaxis']['title'] = "最適化された評価値"
    # plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    # sort_idx = np.argsort(val_loss_concat)[::-1]
    # x = [str(idx) for idx in sort_idx]
    # y = np.array(val_loss_concat)[sort_idx]
    # fig = go.Figure()
    # fig.add_trace(go.Bar(
    #     x = x, y = y,
    # ))
    # fig['layout']['xaxis']['title'] = "data index"
    # fig['layout']['yaxis']['title'] = "validation error"
    # plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)
    
    
def gmm_test12v3():
    n_samples = 100
    
    with open(BASE_DIR + "opttest/gmm_test/shake.pickle", "rb") as f:
        samples = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/shake/GMM12_positive_ridge/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    time_concat = []
    for i in range(n_samples):
    # for i in range(90,100):
        if len(samples["jpx"][i]) == 0:
            train_loss_concat.append(0)
            val_loss_concat.append(0)
            continue
        jpx, uq_index = np.unique(samples["jpx"][i], return_index = True)
        jpy = samples["jpy"][i][uq_index]
        func = samples["func"][i]
        p = samples["p"][i]
        
        Var = np.diag([(0.8-0.3)/(100./p)])**2
        # gmm = GMM8(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 100, 'verbose': 0})
        # gmm = GMM12(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0})
        gmm = GMM9(None, diag_sigma = np.sqrt(Var), options = {"tau": 0.9, "lam": 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        t1 = time.time()
        min_loss = gmm.train(recreate_jp = False)
        t2 = time.time() - t1
        print(i, "Time: ", t2)
        print(i, "Loss: ", min_loss)
        time_concat.append(t2)
        train_loss_concat.append(min_loss)
        
        xlist = np.linspace(0.3,0.8,500)
        ylist = np.array([func(x) for x in xlist])
        plist = gmm.predict(xlist.reshape(-1,1))
        val_loss = np.mean(np.abs(ylist - plist))
        val_loss_concat.append(val_loss)
        
        # fig = plt.figure()
        # plt.scatter(jpx, jpy, color="orange")
        # plt.plot(xlist, ylist, color="skyblue")
        # plt.plot(xlist, plist, color="purple")
        # plt.savefig(save_img_dir+"t{}.png".format(i))
        # plt.close()
        
    print("Time mean: ", np.mean(time_concat))
    print("Time sd: ", np.std(time_concat))
    print("Train Loss mean: ", np.mean(train_loss_concat))
    print("Train Loss sd: ", np.std(train_loss_concat))
    print("Validation Loss mean: ", np.mean(val_loss_concat))
    print("Validation Loss sd: ", np.std(val_loss_concat))
        
    # sort_idx = np.argsort(train_loss_concat)[::-1]
    # x = [str(idx) for idx in sort_idx]
    # y = np.array(train_loss_concat)[sort_idx]
    # fig = go.Figure()
    # fig.add_trace(go.Bar(
    #     x = x, y = y,
    # ))
    # fig['layout']['xaxis']['title'] = "data index"
    # fig['layout']['yaxis']['title'] = "最適化された評価値"
    # plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    # sort_idx = np.argsort(val_loss_concat)[::-1]
    # x = [str(idx) for idx in sort_idx]
    # y = np.array(val_loss_concat)[sort_idx]
    # fig = go.Figure()
    # fig.add_trace(go.Bar(
    #     x = x, y = y,
    # ))
    # fig['layout']['xaxis']['title'] = "data index"
    # fig['layout']['yaxis']['title'] = "validation error"
    # plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)


def gmm_test13():
    n_samples = 100
    pref_list = [
        (lambda i: BASE_DIR + "opttest/logs/onpolicy2/GMM9Sig5LCB3/checkpoints/t{}/ch500/".format(i), 5),
        (lambda i: BASE_DIR + "opttest/logs/onpolicy2/GMM9Sig8LCB4/checkpoints/t{}/ch500/".format(i), 8),
    ]
    
    samples = {"jpx": [], "jpy": [], "func": [], "logdir": [], "p": []}
    for _ in range(n_samples):
        i = np.random.randint(1,99)
        pref, p = pref_list[RandI(len(pref_list))]
        logdir = pref(i)
        print(_, logdir)
        with open(logdir+"log.yaml", "r") as yml:
            log = yaml.load(yml)
        j = np.random.randint(50,501)
        jpx = np.array(log["est_gmm_JP_tip"][j]["X"])
        jpy = np.array(log["est_gmm_JP_tip"][j]["Y"])
        Var = np.diag([(1.0-0.1)/(100./p),(0.8-0.3)/(100./p)])**2
        func = lambda x, jpx=jpx, jpy=jpy, Var=Var: [multivariate_normal.pdf(x,jpx_i,Var)*(1./multivariate_normal.pdf(jpx_i,jpx_i,Var))*jpy_i for jpx_i,jpy_i in zip(jpx, jpy)]
        samples["jpx"].append(jpx)
        samples["jpy"].append(jpy)
        samples["func"].append(func)
        samples["p"].append(p)
        samples["logdir"].append(logdir+"_"+str(j))

    savedir = BASE_DIR + "opttest/gmm_test/"
    check_or_create_dir(savedir)
    with open(savedir+"tip.pickle", mode="wb") as f:
        dill.dump(samples, f)
        
        
def gmm_test14():
    n_samples = 100
    
    with open(BASE_DIR + "opttest/gmm_test/tip.pickle", "rb") as f:
        samples = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/tip/GMM8_positive_ridge/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    time_concat = []
    for i in range(n_samples):
    # for i in [10,15,80,25,45,17,19,48]:
    #     print(samples["logdir"][i], len(samples["jpx"][i]))
    #     continue
        if len(samples["jpx"][i]) == 0:
            train_loss_concat.append(0)
            val_loss_concat.append(0)
            continue
        jpx, uq_index = np.unique(samples["jpx"][i], axis = 0, return_index = True)
        jpy = samples["jpy"][i][uq_index]
        func = samples["func"][i]
        p = samples["p"][i]
        
        Var = np.diag([(1.0-0.1)/(100./p),(0.8-0.3)/(100./p)])**2
        gmm = GMM8(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 100, 'verbose': 0})
        # gmm = GMM8(None, diag_sigma = np.sqrt(Var), w_positive = False, options = {"tau": 0.9, "lam": 0, 'maxiter': 1e2, 'popsize': 500, 'verbose': 0})
        # gmm = GMM9(None, diag_sigma=np.sqrt(Var), options = {"tau": 0.9, "lam": 0})
        # gmm = GMM9v2(None, diag_sigma=np.sqrt(Var), options = {"tau": 0.9, "lam": 0})
        # gmm = GMM11(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 100, 'verbose': 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        t1 = time.time()
        min_loss = gmm.train(recreate_jp = False)
        t2 = time.time() - t1
        print(i, "Time: ", t2)
        print(i, "Loss: ", min_loss)
        time_concat.append(t2)
        train_loss_concat.append(min_loss)
        # print(gmm.w_concat)
        
        X = np.array([[dtheta2, smsz] for dtheta2 in np.linspace(0.1,1.0,100) for smsz in np.linspace(0.3,0.8,100)])
        P = gmm.predict(X).reshape(100,100)
        Y = (np.max(func(X), axis = 0)).reshape(100,100)
        val_loss = np.mean(np.abs(Y - P))
        val_loss_concat.append(val_loss)
        
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=["飛び値 & 混合するガウシアンのmax値", "飛び値モデル予測"],
            specs = [[{"type": "surface"}, {"type": "surface"}]],
        )
        fig.update_layout(
            height=1000, width=2000, 
            hoverdistance = 2,
        )
        fig.add_trace(go.Scatter3d(
            z = jpy.flatten(), x = jpx[:,1].flatten(), y = jpx[:,0].flatten(),
            mode = "markers",
            marker = dict(
                color = "red",
                size = 4,
            ),
            showlegend = False,
        ), 1, 1)
        fig.add_trace(go.Surface(
            z = Y, x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1.0,100),
            cmin = 0, cmax = 0.2, 
            colorbar = dict(len = 0.4),
            colorscale = [
                [0, "rgb(0, 0, 0)"],
                [0.01, "rgba(3, 0, 200, 0.4)"],
                [0.1, "rgba(60, 50, 160, 0.6)"],
                [0.2, "rgba(30, 50, 120, 0.4)"],
                [0.4, "rgba(0, 100, 100, 0.4)"],
                [0.6, "rgba(0, 150, 40, 0.4)"],
                [0.8, "rgba(0, 200, 20, 0.3)"],
                [1, "rgba(0, 255, 0, 0.3)"],
            ],
            showlegend = False,
        ), 1, 1)
        fig.add_trace(go.Scatter3d(
            z = jpy.flatten(), x = jpx[:,1].flatten(), y = jpx[:,0].flatten(),
            mode = "markers",
            marker = dict(
                color = "red",
                size = 4,
            ),
            showlegend = False,
        ), 1, 2)
        fig.add_trace(go.Surface(
            z = P, x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1.0,100),
            cmin = 0, cmax = 0.2, 
            colorbar = dict(len = 0.4),
            colorscale = [
                [0, "rgb(0, 0, 0)"],
                [0.01, "rgba(3, 0, 200, 0.4)"],
                [0.1, "rgba(60, 50, 160, 0.6)"],
                [0.2, "rgba(30, 50, 120, 0.4)"],
                [0.4, "rgba(0, 100, 100, 0.4)"],
                [0.6, "rgba(0, 150, 40, 0.4)"],
                [0.8, "rgba(0, 200, 20, 0.3)"],
                [1, "rgba(0, 255, 0, 0.3)"],
            ],
            showlegend = False,
        ), 1, 2)
        fig['layout']['scene']['xaxis']['title'] = "size_srcmouth" 
        fig['layout']['scene']['yaxis']['title'] = "dtheta2" 
        fig['layout']['scene']['zaxis']['title'] = "estimation / true value"
        plotly.offline.plot(fig, filename = save_img_dir+"t{}.html".format(i), auto_open=False)
    
    print("Time mean: ", np.mean(time_concat))
    print("Time sd: ", np.std(time_concat))
    print("Train Loss mean: ", np.mean(train_loss_concat))
    print("Train Loss sd: ", np.std(train_loss_concat))
    print("Validation Loss mean: ", np.mean(val_loss_concat))
    print("Validation Loss sd: ", np.std(val_loss_concat))
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)
    

def gmm_test15():
    n_samples = 100
    
    with open(BASE_DIR + "opttest/gmm_test/tip.pickle", "rb") as f:
        samples = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/tip/GMM12_positive_ridge/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    time_concat = []
    for i in range(n_samples):
    # for i in [10,15,80,25,45,17,19,48]:
    #     print(samples["logdir"][i], len(samples["jpx"][i]))
    #     continue
        if len(samples["jpx"][i]) == 0:
            train_loss_concat.append(0)
            val_loss_concat.append(0)
            continue
        jpx, uq_index = np.unique(samples["jpx"][i], axis = 0, return_index = True)
        jpy = samples["jpy"][i][uq_index]
        func = samples["func"][i]
        p = samples["p"][i]
        
        Var = np.diag([(1.0-0.1)/(100./p),(0.8-0.3)/(100./p)])**2
        # gmm = GMM8(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 100, 'verbose': 0})
        gmm = GMM12(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        t1 = time.time()
        min_loss = gmm.train(recreate_jp = False)
        t2 = time.time() - t1
        print(i, "Time: ", t2)
        print(i, "Loss: ", min_loss)
        time_concat.append(t2)
        train_loss_concat.append(min_loss)
        # print(gmm.w_concat)
        
        X = np.array([[dtheta2, smsz] for dtheta2 in np.linspace(0.1,1.0,100) for smsz in np.linspace(0.3,0.8,100)])
        P = gmm.predict(X).reshape(100,100)
        Y = (np.max(func(X), axis = 0)).reshape(100,100)
        val_loss = np.mean(np.abs(Y - P))
        val_loss_concat.append(val_loss)
        
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=["飛び値 & 混合するガウシアンのmax値", "飛び値モデル予測"],
            specs = [[{"type": "surface"}, {"type": "surface"}]],
        )
        fig.update_layout(
            height=1000, width=2000, 
            hoverdistance = 2,
        )
        fig.add_trace(go.Scatter3d(
            z = jpy.flatten(), x = jpx[:,1].flatten(), y = jpx[:,0].flatten(),
            mode = "markers",
            marker = dict(
                color = "red",
                size = 4,
            ),
            showlegend = False,
        ), 1, 1)
        fig.add_trace(go.Surface(
            z = Y, x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1.0,100),
            cmin = 0, cmax = 0.2, 
            colorbar = dict(len = 0.4),
            colorscale = [
                [0, "rgb(0, 0, 0)"],
                [0.01, "rgba(3, 0, 200, 0.4)"],
                [0.1, "rgba(60, 50, 160, 0.6)"],
                [0.2, "rgba(30, 50, 120, 0.4)"],
                [0.4, "rgba(0, 100, 100, 0.4)"],
                [0.6, "rgba(0, 150, 40, 0.4)"],
                [0.8, "rgba(0, 200, 20, 0.3)"],
                [1, "rgba(0, 255, 0, 0.3)"],
            ],
            showlegend = False,
        ), 1, 1)
        fig.add_trace(go.Scatter3d(
            z = jpy.flatten(), x = jpx[:,1].flatten(), y = jpx[:,0].flatten(),
            mode = "markers",
            marker = dict(
                color = "red",
                size = 4,
            ),
            showlegend = False,
        ), 1, 2)
        fig.add_trace(go.Surface(
            z = P, x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1.0,100),
            cmin = 0, cmax = 0.2, 
            colorbar = dict(len = 0.4),
            colorscale = [
                [0, "rgb(0, 0, 0)"],
                [0.01, "rgba(3, 0, 200, 0.4)"],
                [0.1, "rgba(60, 50, 160, 0.6)"],
                [0.2, "rgba(30, 50, 120, 0.4)"],
                [0.4, "rgba(0, 100, 100, 0.4)"],
                [0.6, "rgba(0, 150, 40, 0.4)"],
                [0.8, "rgba(0, 200, 20, 0.3)"],
                [1, "rgba(0, 255, 0, 0.3)"],
            ],
            showlegend = False,
        ), 1, 2)
        fig['layout']['scene']['xaxis']['title'] = "size_srcmouth" 
        fig['layout']['scene']['yaxis']['title'] = "dtheta2" 
        fig['layout']['scene']['zaxis']['title'] = "estimation / true value"
        plotly.offline.plot(fig, filename = save_img_dir+"t{}.html".format(i), auto_open=False)
    
    print("Time mean: ", np.mean(time_concat))
    print("Time sd: ", np.std(time_concat))
    print("Train Loss mean: ", np.mean(train_loss_concat))
    print("Train Loss sd: ", np.std(train_loss_concat))
    print("Validation Loss mean: ", np.mean(val_loss_concat))
    print("Validation Loss sd: ", np.std(val_loss_concat))
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)


def gmm_test15v2():
    n_samples = 100
    
    with open(BASE_DIR + "opttest/gmm_test/tip.pickle", "rb") as f:
        samples = dill.load(f)
    save_img_dir = PICTURE_DIR + "opttest/gmm_test/tip/GMM12_positive_ridge_v2/"
    check_or_create_dir(save_img_dir)
    
    train_loss_concat = []
    val_loss_concat = []
    time_concat = []
    for i in range(n_samples):
    # for i in [10,15,80,25,45,17,19,48]:
    #     print(samples["logdir"][i], len(samples["jpx"][i]))
    #     continue
        if len(samples["jpx"][i]) == 0:
            train_loss_concat.append(0)
            val_loss_concat.append(0)
            continue
        jpx, uq_index = np.unique(samples["jpx"][i], axis = 0, return_index = True)
        jpy = samples["jpy"][i][uq_index]
        func = samples["func"][i]
        p = samples["p"][i]
        
        Var = np.diag([(1.0-0.1)/(100./p),(0.8-0.3)/(100./p)])**2
        # gmm = GMM8(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 100, 'verbose': 0})
        # gmm = GMM12(None, diag_sigma = np.sqrt(Var), w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 5e-4, 'verbose': 0})
        gmm = GMM9(None, diag_sigma = np.sqrt(Var), options = {"tau": 0.9, "lam": 0})
        gmm.jumppoints.update({"X": jpx, "Y": jpy})
        t1 = time.time()
        min_loss = gmm.train(recreate_jp = False)
        t2 = time.time() - t1
        print(i, "Time: ", t2)
        print(i, "Loss: ", min_loss)
        time_concat.append(t2)
        train_loss_concat.append(min_loss)
        # print(gmm.w_concat)
        
        X = np.array([[dtheta2, smsz] for dtheta2 in np.linspace(0.1,1.0,100) for smsz in np.linspace(0.3,0.8,100)])
        P = gmm.predict(X).reshape(100,100)
        Y = (np.max(func(X), axis = 0)).reshape(100,100)
        val_loss = np.mean(np.abs(Y - P))
        val_loss_concat.append(val_loss)
        
        # fig = make_subplots(
        #     rows=1, cols=2, 
        #     subplot_titles=["飛び値 & 混合するガウシアンのmax値", "飛び値モデル予測"],
        #     specs = [[{"type": "surface"}, {"type": "surface"}]],
        # )
        # fig.update_layout(
        #     height=1000, width=2000, 
        #     hoverdistance = 2,
        # )
        # fig.add_trace(go.Scatter3d(
        #     z = jpy.flatten(), x = jpx[:,1].flatten(), y = jpx[:,0].flatten(),
        #     mode = "markers",
        #     marker = dict(
        #         color = "red",
        #         size = 4,
        #     ),
        #     showlegend = False,
        # ), 1, 1)
        # fig.add_trace(go.Surface(
        #     z = Y, x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1.0,100),
        #     cmin = 0, cmax = 0.2, 
        #     colorbar = dict(len = 0.4),
        #     colorscale = [
        #         [0, "rgb(0, 0, 0)"],
        #         [0.01, "rgba(3, 0, 200, 0.4)"],
        #         [0.1, "rgba(60, 50, 160, 0.6)"],
        #         [0.2, "rgba(30, 50, 120, 0.4)"],
        #         [0.4, "rgba(0, 100, 100, 0.4)"],
        #         [0.6, "rgba(0, 150, 40, 0.4)"],
        #         [0.8, "rgba(0, 200, 20, 0.3)"],
        #         [1, "rgba(0, 255, 0, 0.3)"],
        #     ],
        #     showlegend = False,
        # ), 1, 1)
        # fig.add_trace(go.Scatter3d(
        #     z = jpy.flatten(), x = jpx[:,1].flatten(), y = jpx[:,0].flatten(),
        #     mode = "markers",
        #     marker = dict(
        #         color = "red",
        #         size = 4,
        #     ),
        #     showlegend = False,
        # ), 1, 2)
        # fig.add_trace(go.Surface(
        #     z = P, x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1.0,100),
        #     cmin = 0, cmax = 0.2, 
        #     colorbar = dict(len = 0.4),
        #     colorscale = [
        #         [0, "rgb(0, 0, 0)"],
        #         [0.01, "rgba(3, 0, 200, 0.4)"],
        #         [0.1, "rgba(60, 50, 160, 0.6)"],
        #         [0.2, "rgba(30, 50, 120, 0.4)"],
        #         [0.4, "rgba(0, 100, 100, 0.4)"],
        #         [0.6, "rgba(0, 150, 40, 0.4)"],
        #         [0.8, "rgba(0, 200, 20, 0.3)"],
        #         [1, "rgba(0, 255, 0, 0.3)"],
        #     ],
        #     showlegend = False,
        # ), 1, 2)
        # fig['layout']['scene']['xaxis']['title'] = "size_srcmouth" 
        # fig['layout']['scene']['yaxis']['title'] = "dtheta2" 
        # fig['layout']['scene']['zaxis']['title'] = "estimation / true value"
        # plotly.offline.plot(fig, filename = save_img_dir+"t{}.html".format(i), auto_open=False)
    
    print("Time mean: ", np.mean(time_concat))
    print("Time sd: ", np.std(time_concat))
    print("Train Loss mean: ", np.mean(train_loss_concat))
    print("Train Loss sd: ", np.std(train_loss_concat))
    print("Validation Loss mean: ", np.mean(val_loss_concat))
    print("Validation Loss sd: ", np.std(val_loss_concat))
        
    sort_idx = np.argsort(train_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(train_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "最適化された評価値"
    plotly.offline.plot(fig, filename = save_img_dir+"train_loss.html", auto_open=False)

    sort_idx = np.argsort(val_loss_concat)[::-1]
    x = [str(idx) for idx in sort_idx]
    y = np.array(val_loss_concat)[sort_idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
    ))
    fig['layout']['xaxis']['title'] = "data index"
    fig['layout']['yaxis']['title'] = "validation error"
    plotly.offline.plot(fig, filename = save_img_dir+"val_loss.html", auto_open=False)


def gmm_logcheck():
    logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/TMMSig8LCB4/checkpoints/t76/ch500/"
    with open(logdir+"log.yaml") as f:
        log = yaml.load(f)
    for i in range(397,401):
        print(i, log["smsz"][i], log["est_optparam"][i])


def gmm_ep(name, ep):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/{}/".format(name, ep)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    datotal = setup_datotal(dm, logdir)
    gmm = dm.gmms[TIP]
    gmm.jumppoints = log["est_gmm_JP_tip"][ep]
    gmm.train(recreate_jp = False)
    X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ]).astype(np.float32)
    gmmpred = gmm.predict(X).reshape(100,100)
    
    diffcs = [
        [0, "rgb(0, 0, 0)"],
        [0.01, "rgb(255, 255, 200)"],
        [1, "rgb(255, 0, 0)"],
    ]
    jpx_idx = [[idx_of_the_nearest(dm.dtheta2, x[0]), idx_of_the_nearest(dm.smsz, x[1])] for x in np.array(gmm.jumppoints["X"])]
    jpx_tr = [dm.datotal[TIP][RFUNC][idx[0],idx[1]] for idx in jpx_idx]
    jpx_gmm = [gmmpred[idx[0],idx[1]] for idx in jpx_idx]
    jpy = [y[0] for y in gmm.jumppoints["Y"]]
    linex = [[x,x] for x in np.array(gmm.jumppoints["X"])[:,1]]
    liney = [[y,y] for y in np.array(gmm.jumppoints["X"])[:,0]]
    linegmm =[[a,b] for a, b in zip(jpy, jpx_gmm)]
        
    fig = go.Figure()
    fig.add_trace(go.Surface(
        z = gmmpred, x = dm.smsz, y = dm.dtheta2,
        cmin = 0, cmax = 0.2, colorscale = diffcs,
        showlegend = False,
    ))
    fig.add_trace(go.Scatter3d(
        z = jpy, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
        marker = dict(
            color = "red",
            size = 4,
        )
    ))
    for tz,tx,ty in zip(linegmm, linex, liney):
        fig.add_trace(go.Scatter3d(
            z = tz, x = tx, y = ty,
            mode = "lines",
            line = dict(
                color = "red",
            ),
            showlegend = False,
        ))
    fig['layout']['scene']['xaxis']['title'] = "size_srcmouth" 
    fig['layout']['scene']['yaxis']['title'] = "dtheta2" 
    fig['layout']['scene']['zaxis']['title'] = "gmm predict" 
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "gmm_tip.html", auto_open=False)


class Domain3:
    def __init__(self, logdir, sd_gain = 1.0, LCB_ratio = 0.0, without_smsz = None):
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
            "est_gmm_JP_tip": [],
            "est_gmm_JP_shake": [],
        }
        self.logdir = logdir
        self.rmodel = Rmodel("Fdatotal_gentle")
        self.without_smsz = without_smsz

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
            for skill in [TIP, SHAKE]:
                self.gmms[skill] = gmm_lams[skill](self.nnmodels[skill])
                    
    def optimize(self, smsz):
        ###########################
        ###Tip用最適化
        ###########################
        est_nn_Er, est_nn_Sr = [], []
        if self.use_gmm:
            # self.gmms[TIP].train()
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
        opteval_TIP = evalmatrix[idx_est_optparam]
        
        ###########################
        ###Shake用最適化
        ###########################
        if self.use_gmm:
            # self.gmms[SHAKE].train()
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
            est_datotal = self.nnmodels[SHAKE].model.Predict(x = [smsz], with_var=True).Y[0].item()
            opteval = opteval_SHAKE
        else:
            skill = TIP
            est_datotal = self.nnmodels[TIP].model.Predict(x=[est_optparam, smsz], with_var=True).Y[0].item()
            opteval = opteval_TIP
            
        if self.use_gmm:
            self.gmms[TIP].train()
            self.gmms[SHAKE].train()
        
        return skill, idx_est_optparam, est_optparam, est_datotal, opteval
                    
    def execute_main(self, idx_smsz, smsz, num_rand_sample, num_learn_step):
        ep = len(self.log["ep"])
        print("ep: {}".format(ep))
            
        if ep< num_rand_sample:
            if ep < int(num_rand_sample/2):    #Tip用ランダムサンプリング
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
        
        if ep % num_learn_step == 0:    not_learn = False
        else:                           not_learn = True
        
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
        
    def execute(self, num_rand_sample, num_learn_step):
        idx_smsz = RandI(len(self.smsz))
        # ep = len(self.log["ep"])
        # idx_smsz = ep%100
        smsz = self.smsz[idx_smsz]
        if self.without_smsz != None:
            while (self.without_smsz[0] < smsz.item() < self.without_smsz[1]):
                idx_smsz = RandI(len(self.smsz))
                smsz = self.smsz[idx_smsz]
        self.execute_main(idx_smsz, smsz, num_rand_sample, num_learn_step)
            
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


def calc_rfunc():
    logpath = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/smsz06065g0/pred_true_log.yaml"
    with open(logpath, "r") as f:
        a = yaml.load(f)
    r_shake_crop = np.array([rfunc(a[i]["Fshake_amount"]["true_output"][0]) for i in range(len(a))])
    y_shake_crop = np.array([a[i]["Fshake_amount"]["true_output"][0] for i in range(len(a))])
    # y_shake_crop = np.random.normal(0.4,0.1,100)
    r_shake_crop = np.array([rfunc(y) for y in y_shake_crop])
    logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/Er/t1/"
    dm = Domain3.load(logdir+"dm.pickle")
    rmodel = Rmodel("Fdatotal_gentle")
    r_tip = dm.datotal[TIP][RFUNC]
    r_tip_crop = np.array(r_tip[60:])[:,63:68]
    y_tip_crop = np.array(dm.datotal[TIP][TRUE][60:])[:,63:68]
    # r_tip_crop = r_tip_crop[r_tip_crop>-0.252]
    
    # for p in [5,25,50,75,95]:
    #     print(p, np.percentile(r_tip_crop, p))
    #     print(p, np.percentile(r_shake_crop, p))
    # for p in [5,25,50,75,95]:
    #     print(p, np.percentile(y_tip_crop, p))
    #     print(p, np.percentile(y_shake_crop, p))
    print(np.mean(y_tip_crop), np.std(y_tip_crop))
    print(np.mean(y_shake_crop), np.std(y_shake_crop))
    r_tip = rmodel.Predict(x = [0.3, np.mean(y_tip_crop)], x_var = [0, np.std(y_tip_crop)**2], with_var = True)
    # r_tip = rmodel.Predict(x = [0.3, np.mean(y_tip_crop)], x_var = [0, 0.12**2], with_var = True)
    r_shake = rmodel.Predict(x = [0.3, np.mean(y_shake_crop)], x_var = [0, np.std(y_shake_crop)**2], with_var = True)
    print(r_tip.Y.item(), np.sqrt(r_tip.Var).item())
    print(r_shake.Y.item(), np.sqrt(r_shake.Var).item())
    
    
    print(np.mean(r_tip_crop), np.std(r_tip_crop))
    print(np.mean(r_shake_crop), np.std(r_shake_crop))
    # for _ in range(10):
    #     a = random.sample(r_tip_crop, 10)
    #     print(np.mean(a), np.std(a))
    # print(np.mean(r_tip_crop[r_tip_crop>-0.343]), np.std(r_tip_crop[r_tip_crop>-0.343]))
    # print(np.mean(r_shake_crop[r_shake_crop>-0.895]), np.std(r_shake_crop[r_shake_crop>-0.895]))
    # print(np.mean(r_tip_crop[r_tip_crop>-1]), np.std(r_tip_crop[r_tip_crop>-1]))
    # a = (r_tip_crop[r_tip_crop>-1]).tolist()
    # for _ in range(3):
    #     a.append(-4)
    # print(np.std(a))
    # print(np.sort(r_shake_crop))
    # print(dm.dtheta2[60])
    
    
    # fig = go.Figure()
    # fig.add_trace(go.Box(
    #     # x0 = np.array(dm.smsz)[j]+0.001*n_i, 
    #     x0 = "Tip (B)",
    #             upperfence = [np.percentile(r_tip_crop, 95)],
    #             q3 = [np.percentile(r_tip_crop, 75)],
    #             median = [np.percentile(r_tip_crop, 50)],
    #             q1 = [np.percentile(r_tip_crop, 25)],
    #             lowerfence = [np.percentile(r_tip_crop, 5)],            
    #             fillcolor = "white",
    #             marker = dict(color = "blue", line = dict(outliercolor = "blue")),
    #             # width = 0.0015,
    #             # whiskerwidth = 1,
    #             showlegend = False,
    # ))
    # fig.add_trace(go.Box(
    #     # x0 = np.array(dm.smsz)[j]+0.001*n_i, 
    #     x0 = "Shake (B)",
    #             upperfence = [np.percentile(r_shake_crop, 95)],
    #             q3 = [np.percentile(r_shake_crop, 75)],
    #             median = [np.percentile(r_shake_crop, 50)],
    #             q1 = [np.percentile(r_shake_crop, 25)],
    #             lowerfence = [np.percentile(r_shake_crop, 5)],            
    #             fillcolor = "white",
    #             marker = dict(color = "red", line = dict(outliercolor = "red")),
    #             # width = 0.0015,
    #             # whiskerwidth = 1,
    #             showlegend = False,
    # ))
    # fig.show()
    
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        # x0 = np.array(dm.smsz)[j]+0.001*n_i, 
        x0 = "Tip",
                upperfence = [np.percentile(y_tip_crop, 95)],
                q3 = [np.percentile(y_tip_crop, 75)],
                median = [np.percentile(y_tip_crop, 50)],
                q1 = [np.percentile(y_tip_crop, 25)],
                lowerfence = [np.percentile(y_tip_crop, 5)],         
                # mean = [np.mean(y_tip_crop)],
                # sd = [np.std(y_tip_crop)],
                fillcolor = "white",
                marker = dict(color = "blue", line = dict(outliercolor = "blue")),
                # width = 0.0015,
                # whiskerwidth = 1,
                showlegend = False,
    ))
    fig.add_trace(go.Scatter(
        x = ["Tip"], y = [np.mean(y_tip_crop)],
        error_y = dict(
            symmetric = True,
            array = [np.std(y_tip_crop)],
            width = 20,
            color = "red",
            thickness = 5,
        ),
        mode = "markers",
        marker = dict(
            color = "red",
            symbol = "x",
            size = 10,
        ),
    ))
    fig.add_trace(go.Box(
        # x0 = np.array(dm.smsz)[j]+0.001*n_i, 
        x0 = "Shake",
                upperfence = [np.percentile(y_shake_crop, 95)],
                q3 = [np.percentile(y_shake_crop, 75)],
                median = [np.percentile(y_shake_crop, 50)],
                q1 = [np.percentile(y_shake_crop, 25)],
                lowerfence = [np.percentile(y_shake_crop, 5)],
                # mean = [np.mean(y_shake_crop)],
                # sd = [np.std(y_shake_crop)],       
                fillcolor = "white",
                marker = dict(color = "blue", line = dict(outliercolor = "blue")),
                # width = 0.0015,
                # whiskerwidth = 1,
                showlegend = False,
    ))
    fig.add_trace(go.Scatter(
        x = ["Shake"], y = [np.mean(y_shake_crop)],
        error_y = dict(
            symmetric = True,
            array = [np.std(y_shake_crop)],
            width = 20,
            color = "red",
            thickness = 5,
        ),
        mode = "markers",
        marker = dict(
            color = "red",
            symbol = "x",
            size = 10,
        ),
    ))
    fig.show()
    
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        # x0 = np.array(dm.smsz)[j]+0.001*n_i, 
        x0 = "Tip",
                upperfence = [np.percentile(r_tip_crop, 95)],
                q3 = [np.percentile(r_tip_crop, 75)],
                median = [np.percentile(r_tip_crop, 50)],
                q1 = [np.percentile(r_tip_crop, 25)],
                lowerfence = [np.percentile(r_tip_crop, 5)],         
                mean = [np.mean(r_tip_crop)],
                sd = [np.std(r_tip_crop)],
                fillcolor = "white",
                marker = dict(color = "blue", line = dict(outliercolor = "blue")),
                # width = 0.0015,
                # whiskerwidth = 1,
                showlegend = False,
    ))
    fig.add_trace(go.Scatter(
        x = ["Tip"], y = [r_tip.Y.item()],
        error_y = dict(
            symmetric = True,
            array = [np.sqrt(r_tip.Var.item())],
            width = 20,
            color = "red",
            thickness = 5,
        ),
        mode = "markers",
        marker = dict(
            color = "red",
            symbol = "x",
            size = 10,
        ),
    ))
    fig.add_trace(go.Scatter(
        x = ["Tip (Wolfram Alpha)"], y = [-0.1058],
        error_y = dict(
            symmetric = True,
            array = [0.2747],
            width = 20,
            color = "purple",
            thickness = 5,
        ),
        mode = "markers",
        marker = dict(
            color = "purple",
            symbol = "x",
            size = 10,
        ),
    ))
    fig.add_trace(go.Box(
        # x0 = np.array(dm.smsz)[j]+0.001*n_i, 
        x0 = "Shake",
                upperfence = [np.percentile(r_shake_crop, 95)],
                q3 = [np.percentile(r_shake_crop, 75)],
                median = [np.percentile(r_shake_crop, 50)],
                q1 = [np.percentile(r_shake_crop, 25)],
                lowerfence = [np.percentile(r_shake_crop, 5)],
                mean = [np.mean(r_shake_crop)],
                sd = [np.std(r_shake_crop)],       
                fillcolor = "white",
                marker = dict(color = "blue", line = dict(outliercolor = "blue")),
                # width = 0.0015,
                # whiskerwidth = 1,
                showlegend = False,
    ))
    fig.add_trace(go.Scatter(
        x = ["Shake"], y = [r_shake.Y.item()],
        error_y = dict(
            symmetric = True,
            array = [np.sqrt(r_shake.Var.item())],
            width = 20,
            color = "red",
            thickness = 5,
        ),
        mode = "markers",
        marker = dict(
            color = "red",
            symbol = "x",
            size = 10,
        ),
    ))
    fig.add_trace(go.Scatter(
        x = ["Shake (Woflfram Alpha)"], y = [-0.7084],
        error_y = dict(
            symmetric = True,
            array = [0.214],
            width = 20,
            color = "purple",
            thickness = 5,
        ),
        mode = "markers",
        marker = dict(
            color = "purple",
            symbol = "x",
            size = 10,
        ),
    ))
    fig.add_shape(
        type = "line",
        x0 = -0.5, x1 = 3.5,
        y0 = 0, y1 = 0,
    )
    fig.show()
        

def shake_rfunc_plot():
    datotal = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal.npy")
    r = []
    for d in datotal:
        # r.append(rfunc(d))
        r.append(d)
    # fig = plt.figure()
    # plt.scatter(x = np.linspace(0.3,0.8,100), y = r)
    # # plt.hlines(xmin=0.3,xmax=0.8,y=-1,color="red",linestyle="dashed")
    # # plt.hlines(xmin=0.3,xmax=0.8,y=0.3,color="red",linestyle="dashed")
    # plt.xticks(fontsize = 16)
    # plt.yticks(fontsize = 16)
    # plt.ylim(0.28,0.57)
    # plt.subplots_adjust(left=0.05, right=0.98)
    # plt.show()
    
    r = np.array(r).reshape(1,-1)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z = r, x = np.linspace(0.3,0.8,100), y = [1],
        colorscale = [
        [0, "rgb(0, 0, 255)"],
        [0.2727, "rgb(0, 255, 255)"],
        [0.5454, "rgb(0, 255, 0)"],
        [0.772, "rgb(255, 255, 0)"],
        [1, "rgb(255, 0, 0)"],
    ],
        zmin = 0, zmax = 0.55,
    ))
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 60
    fig.show()
    
    # fig = plt.figure()
    # datotals = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal2.npy")
    # R = []
    # for datotal in datotals[0:2]:
    #     r = []
    #     for d in datotal:
    #         r.append(rfunc(d))
    #     R.append(r)
    # R = np.array(R)
    # plt.scatter(x = np.linspace(0.3,0.8,100), y = R.max(axis=0))
    # plt.show
    
    # datotals = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal2.npy")
    # for datotal, rng in zip(datotals, [0.8,0.7,0.6,0.5,0.4,0.2,0.1]):
    #     r = []
    #     for d in datotal:
    #         r.append(rfunc(d))
    #     fig = plt.figure()
    #     plt.scatter(x = np.linspace(0.3,0.8,100), y = r)
    #     plt.show()
        
    # datotals = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal3.npy")
    # for datotal, rng in zip(datotals, [0.8,0.7,0.6,0.5,0.4,0.2,0.1]):
    #     r = []
    #     for d in datotal:
    #         r.append(rfunc(d))
    #     fig = plt.figure()
    #     plt.scatter(x = np.linspace(0.3,0.8,100), y = r)
    #     plt.show()


def execute(logdir, sd_gain, LCB_ratio, gmm_lams, num_ep, num_rand_sample, num_learn_step, num_save_ep):
    if os.path.exists(logdir+"dm.pickle"):
        dm = Domain3.load(logdir+"dm.pickle")
    else:
        dm = Domain3(logdir, sd_gain, LCB_ratio)
        dm.setup(gmm_lams)
    
    while len(dm.log["ep"]) < num_ep:
        dm.execute(num_rand_sample = num_rand_sample, num_learn_step = num_learn_step)
        if len(dm.log["ep"])%num_save_ep==0:
            dm.save()
            
            
def execute_checkpoint(base_logdir, sd_gain, LCB_ratio, without_smsz, gmm_lams, num_ep, num_rand_sample, num_learn_step, num_checkpoints):
    ep_checkpoints = [num_ep/num_checkpoints*i for i in range(1,num_checkpoints+1)]
    for ep_checkpoint in ep_checkpoints:
        new_logdir = base_logdir + "ch{}/".format(ep_checkpoint)
        prev_logdir = base_logdir + "ch{}/".format(ep_checkpoint - num_ep/num_checkpoints)
        os.makedirs(new_logdir)
        if os.path.exists(prev_logdir+"dm.pickle"):
            shutil.copytree(prev_logdir+"models", new_logdir+"models")
            dm = Domain3.load(prev_logdir+"dm.pickle")
            dm.logdir = new_logdir
        else:
            dm = Domain3(new_logdir, sd_gain, LCB_ratio, without_smsz)
        dm.setup(gmm_lams)
        
        while len(dm.log["ep"]) < ep_checkpoint:
            dm.execute(num_rand_sample = num_rand_sample, num_learn_step = num_learn_step)
        dm.save()
            
            
def execute_update(ref_logdir, new_logdir, gmm_lams, num_ep, num_rand_sample, num_learn_step, num_save_ep):
    os.mkdir(new_logdir)
    shutil.copytree(ref_logdir+"models", new_logdir+"models")
    
    dm = Domain3.load(ref_logdir+"dm.pickle")
    dm.logdir = new_logdir
    dm.setup(gmm_lams)
    
    while len(dm.log["ep"]) < num_ep:
        dm.execute(num_rand_sample = num_rand_sample, num_learn_step = num_learn_step)
        if len(dm.log["ep"])%num_save_ep==0:
            dm.save()
    
        
def test():
    #Domain用パラメータ
    base_logdir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/"
    name = "test"
    p = 3
    
    execute(**dict(
        num_ep = 25,
        num_rand_sample = 6,
        num_learn_step = 1,
        num_save_ep = 1,
        
        sd_gain = 1.0,
        LCB_ratio = 0.0,
        gmm_lams = {
            TIP: lambda nnmodel: GMM7(nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], Gerr = 1.0),
            SHAKE: lambda nnmodel: GMM7(nnmodel, diag_sigma=[(0.8-0.3)/(100./p)], Gerr = 1.0)
        },
        
        logdir = base_logdir + "logs/onpolicy2/{}/".format(name),
    ))


def opttest_baseline():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/baseline/"
    check_or_create_dir(save_img_dir)
    
    logdir = basedir + "Er/t1/"
    dm = Domain3.load(logdir+"dm.pickle")
    eval_tip = np.load(SRC_PATH+"10_3/r_expec.npy")
    datotal_shake_m10 = np.convolve(dm.datotal[SHAKE][TRUE], np.ones(10)/10, mode="same")
    datotal_shake_v10 = np.convolve(dm.datotal[SHAKE][TRUE]**2, np.ones(10)/10, mode="same") - datotal_shake_m10**2
    # for i in [0,1,2,3,4,5,6,7,8,9]:
    for i in range(0,10):
        datotal_shake_m10[i] =  np.convolve(dm.datotal[SHAKE][TRUE], np.ones(i+1)/(i+1), mode="same")[i]
        datotal_shake_v10[i] = np.convolve(dm.datotal[SHAKE][TRUE]**2, np.ones(i+1)/(i+1), mode="same")[i] - datotal_shake_m10[i]**2
    for i in range(0,10):
        datotal_shake_m10[99-i] =  np.convolve(dm.datotal[SHAKE][TRUE], np.ones(i+1)/(i+1), mode="same")[99-i]
        datotal_shake_v10[99-i] = np.convolve(dm.datotal[SHAKE][TRUE]**2, np.ones(i+1)/(i+1), mode="same")[99-i] - datotal_shake_m10[99-i]**2
    eval_shake = [Rmodel("Fdatotal_gentle").Predict(x=[0.3, m], x_var = [0, v]).Y.item() for m, v in zip(datotal_shake_m10, datotal_shake_v10)]
    r_at_optparam = [dm.datotal[TIP][RFUNC][np.argmax(e_tip),i] if np.max(e_tip) > e_shake else dm.datotal[SHAKE][RFUNC][i] for i, (e_tip, e_shake) in enumerate(zip(eval_tip.T, eval_shake))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = np.max(eval_tip, axis = 0),
        mode = "lines",
        name = "Optimized Tip evaluation",
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = eval_shake,
        mode = "lines",
        name = "Shake evaluation",
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = r_at_optparam,
        mode = "markers",
        name = "True r at optimized parameter",
    ))
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
    )
    fig['layout']['yaxis']['range'] = (-5,0.5)
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    plotly.offline.plot(fig, filename = save_img_dir + "opttest_comp.html", auto_open=False)


def opttest_comp(name, n, ch = None, ver = 2):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/opttest_comp/{}/".format(ver, name, ch)
    check_or_create_dir(save_img_dir)
    
    y_concat = []
    yest_concat = {TIP: [], SHAKE: []}
    for i in range(1,n):
    # for i in range(1,25)+range(26,80)+range(90,96)+range(98,100):
        logdir = basedir + "{}/t{}/{}".format(name, i, ch)
        print(logdir)
        dm = Domain3.load(logdir+"dm.pickle")
        datotal = setup_datotal(dm, logdir)
        gmmpred = setup_gmmpred(dm, logdir)
        evaluation = setup_eval(dm, logdir)
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        est_ytip = np.max(evaluation[TIP], axis=0)
        est_yshake = evaluation[SHAKE]
        
        y = []
        yest = {TIP: [], SHAKE: []}
        for idx_smsz in range(len(dm.smsz)):
            if est_ytip[idx_smsz] > est_yshake[idx_smsz]:
                y.append(true_ytip[idx_smsz])
            else:
                y.append(true_yshake[idx_smsz])
            yest[TIP].append(est_ytip[idx_smsz])
            yest[SHAKE].append(est_yshake[idx_smsz])
        y_concat.append(y)
        for skill in [TIP, SHAKE]:
            yest_concat[skill].append(yest[skill])
    ymean = np.mean(y_concat, axis = 0)
    ysd = np.std(y_concat, axis = 0)
    yestmean = dict()
    yestsd = dict()
    yp = dict()
    yestp = defaultdict(lambda: dict())
    for skill in [TIP, SHAKE]:
        yestmean[skill] = np.mean(yest_concat[skill], axis = 0)
        yestsd[skill] = np.std(yest_concat[skill], axis = 0)
    for p in [0,2,5,10,50,90,95,98,100,25,75]:
        yp[p] = np.percentile(y_concat, p, axis = 0)
        
        # if p in [5,10,50,90,95]:
        #     Print('{}percentile:'.format(p))
        #     for i in range(5):    
        #         Print('smsz_idx {}~{}:'.format(20*i,20*(i+1)), np.mean(yp[p][20*i:20*(i+1)]))
        
        for skill in[TIP, SHAKE]:
            yestp[skill][p] = np.percentile(np.array(yest_concat[skill]), p, axis = 0)
    
    fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x = dm.smsz, y = yp[50],
    #     mode = "markers",
    #     name = "reward at opt param (0%, 100%)",
    #     error_y=dict(
    #         type="data",
    #         symmetric=False,
    #         array=yp[100]-yp[50],
    #         arrayminus=yp[50]-yp[0],
    #         thickness=0.8,
    #         width=3,
    #     ),
    #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    # ))
    # fig.add_trace(go.Scatter(
    #     x = dm.smsz, y = yp[50],
    #     mode = "markers",
    #     name = "reward at opt param (2%, 98%)",
    #     error_y=dict(
    #         type="data",
    #         symmetric=False,
    #         array=yp[98]-yp[50],
    #         arrayminus=yp[50]-yp[2],
    #         thickness=1.5,
    #         width=3,
    #     ),
    #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    # ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yp[50],
        mode = "markers",
        name = "reward at opt param (5%, 50%, 95%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yp[95]-yp[50],
            arrayminus=yp[50]-yp[5],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    badr = [len([yi for yi in y if yi < true_yshake[idx_smsz]]) for idx_smsz, y in enumerate(np.array(y_concat).T)]
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = np.ones(len(dm.smsz))*0.1+np.array([0.1 if i%2==0 else 0 for i in range(len(dm.smsz))]),
        mode = "lines+text",
        text = ["{:.0f}".format(1.*b/len(y_concat)*100) if b != 0 else "" for b in badr],
        line = dict(width=0),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = ymean,
        mode = "markers",
        name = "reward at opt param",
        error_y=dict(
            type="data",
            symmetric=False,
            array=np.zeros(len(ysd)),
            arrayminus=ysd,
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestmean[TIP],
        mode = "markers",
        name = "evaluation (TIP) at opt param",
        error_y=dict(
            type="data",
            symmetric=True,
            array=yestsd[TIP],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestmean[SHAKE],
        mode = "markers",
        name = "evaluation (SHAKE)",
        error_y=dict(
            type="data",
            symmetric=True,
            array=yestsd[SHAKE],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[TIP][50],
        mode = "markers",
        name = "evaluation (TIP) at opt param (0%, 100%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[TIP][100]-yestp[TIP][50],
            arrayminus=yestp[TIP][50]-yestp[TIP][0],
            thickness=0.8,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    # fig.add_trace(go.Scatter(
    #     x = dm.smsz, y = yestp[TIP][50],
    #     mode = "markers",
    #     name = "evaluation (TIP) at opt param (10%, 90%)",
    #     error_y=dict(
    #         type="data",
    #         symmetric=False,
    #         array=yestp[TIP][90]-yestp[TIP][50],
    #         arrayminus=yestp[TIP][50]-yestp[TIP][10],
    #         thickness=0.8,
    #         width=3,
    #     ),
    #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    # ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[TIP][50],
        mode = "markers",
        name = "evaluation (TIP) at opt param (5%, 50%, 95%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[TIP][95]-yestp[TIP][50],
            arrayminus=yestp[TIP][50]-yestp[TIP][5],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[SHAKE][50],
        mode = "markers",
        name = "evaluation (SHAKE) at opt param (0%, 100%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[SHAKE][100]-yestp[SHAKE][50],
            arrayminus=yestp[SHAKE][50]-yestp[SHAKE][0],
            thickness=0.8,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[SHAKE][50],
        mode = "markers",
        name = "evaluation (SHAKE) at opt param (5%, 50%, 95%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[SHAKE][95]-yestp[SHAKE][50],
            arrayminus=yestp[SHAKE][50]-yestp[SHAKE][5],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig['layout']['yaxis']['range'] = (-5,0.5)
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    plotly.offline.plot(fig, filename = save_img_dir + "opttest_comp.html", auto_open=False)
    
    
    fig = go.Figure()
    fig.update_layout(
        height = 600,
    )
    # for n_i, name in enumerate(names[::-1]):
    #     y_concat = y_concat_concat[name]
    #     ymean = ymean_concat[name]
    #     ysd = ysd_concat[name]
    #     yestmean = yestmean_concat[name]
    #     yestsd = yestsd_concat[name]
    #     yp = yp_concat[name]
    #     yestp = yestp_concat[name]
    for i in range(5):
        s_idx, e_idx = 20*i, 20*(i+1)
        fig.add_trace(go.Box(
                # y0 = np.array(dm.smsz)[j] 
                # y0 = n_i,
                # y0 = vis_names[::-1][n_i],
                y0 = '{:.1f}~{:.1f}'.format(dm.smsz[s_idx], dm.smsz[e_idx-1]),
                upperfence = [np.mean(yp[95][s_idx:e_idx])],
                q3 = [np.mean(yp[90][s_idx:e_idx])],
                median = [np.mean(yp[50][s_idx:e_idx])],
                q1 = [np.mean(yp[10][s_idx:e_idx])],
                lowerfence = [np.mean(yp[5][s_idx:e_idx])],            
                # fillcolor = "white",
                marker = dict(color = 'skyblue'),
                # marker = dict(color = colors[::-1][n_i], line = dict(outliercolor = colors[::-1][n_i])),
                line = dict(width = 4),
                width = 0.4,
                # whiskerwidth = 1,
                showlegend = False,
        ))
        fig.add_trace(go.Box(
                # y0 = np.array(dm.smsz)[j] 
                # y0 = n_i,
                # y0 = vis_names[::-1][n_i],
                y0 = '{:.1f}~{:.1f}'.format(dm.smsz[s_idx], dm.smsz[e_idx-1]),
                upperfence = [np.mean(yp[95][s_idx:e_idx])],
                q3 = [np.mean(yp[75][s_idx:e_idx])],
                median = [np.mean(yp[50][s_idx:e_idx])],
                q1 = [np.mean(yp[25][s_idx:e_idx])],
                lowerfence = [np.mean(yp[5][s_idx:e_idx])],            
                fillcolor = "white",
                marker = dict(color = 'blue'),
                # marker = dict(color = colors[::-1][n_i], line = dict(outliercolor = colors[::-1][n_i])),
                line = dict(width = 12),
                width = 0.8,
                # whiskerwidth = 1,
                showlegend = False,
        ))
    # fig['layout']['xaxis']['range'] = (0.295,0.83)
    # fig['layout']['yaxis']['range'] = (-5,0.5)
    # fig['layout']['xaxis']['title'] = "size_srcmouth"
    # fig['layout']['xaxis']['title'] = "s"
    # # fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    # fig['layout']['yaxis']['title'] = "r"
    fig['layout']['xaxis']['range'] = (-2.8,0)
    fig['layout']['xaxis']['title'] = "reward"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
    )
    plotly.offline.plot(fig, filename = save_img_dir + "bar.html", auto_open=False)
    
    
def opttest_comp_concat(names, n, ch = None):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    # basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/offpolicy/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/"
    # save_img_dir = PICTURE_DIR + "opttest/offpolicy/"
    check_or_create_dir(save_img_dir)
    
    names = [
        "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/Er",
        # "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/ErLCB4/checkpoints",
        # "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/GMM12Sig12LCB4/checkpoints",
        # "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/GMM12Sig10LCB4/checkpoints",
        # "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/GMM12Sig8LCB4/checkpoints",
        # "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/GMM12Sig6LCB4/checkpoints",
        # "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/GMM12Sig4LCB4/checkpoints",
        # "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/offpolicy/Er",
        # "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/offpolicy/ErLCB4",
        # "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/offpolicy/GMM12Sig8LCB4",
    ]
    vis_names = [
        "E[r]",
        # "LCB without JPM",
        # "LCB with JPM",
    ]
    chs = [
        "",
        # "ch500/",
        # "ch500/",
        # "ch500/",
        # "ch500/",
        # "ch500/",
        # "ch500/",
        # "ch420/",
        # "ch420/",
        # "ch420/",
    ]
    colors = [
        "green",
        # "blue",
        # "red",
        # "red",
        # "green",
        # "blue",
        # "red",
    ]
    
    y_concat_concat = dict()
    ymean_concat = dict()
    ysd_concat = dict()
    yestmean_concat = dict()
    yestsd_concat = dict()
    yp_concat = dict()
    yestp_concat = dict()
    for name, ch in zip(names, chs):
        y_concat = []
        yest_concat = {TIP: [], SHAKE: []}
        # for i in range(1,n):
        for i in range(2,100):        
        # for i in range(1,25)+range(26,80)+range(90,96)+range(98,100):
            logdir = "{}/t{}/{}".format(name, i, ch)
            print(logdir)
            if not os.path.exists(logdir+"dm.pickle"):
                continue
            dm = Domain3.load(logdir+"dm.pickle")
            datotal = setup_datotal(dm, logdir)
            gmmpred = setup_gmmpred(dm, logdir)
            evaluation = setup_eval(dm, logdir)
            true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
            true_yshake = dm.datotal[SHAKE][RFUNC]
            est_ytip = np.max(evaluation[TIP], axis=0)
            est_yshake = evaluation[SHAKE]
            
            y = []
            yest = {TIP: [], SHAKE: []}
            for idx_smsz in range(len(dm.smsz)):
                if est_ytip[idx_smsz] > est_yshake[idx_smsz]:
                    y.append(true_ytip[idx_smsz])
                else:
                    y.append(true_yshake[idx_smsz])
                yest[TIP].append(est_ytip[idx_smsz])
                yest[SHAKE].append(est_yshake[idx_smsz])
            y_concat.append(y)
            for skill in [TIP, SHAKE]:
                yest_concat[skill].append(yest[skill])
        ymean = np.mean(y_concat, axis = 0)
        ysd = np.std(y_concat, axis = 0)
        yestmean = dict()
        yestsd = dict()
        yp = dict()
        yestp = defaultdict(lambda: dict())
        for skill in [TIP, SHAKE]:
            yestmean[skill] = np.mean(yest_concat[skill], axis = 0)
            yestsd[skill] = np.std(yest_concat[skill], axis = 0)
        for p in [0,2,5,10,25,50,75,90,95,98,100]:
            yp[p] = np.percentile(y_concat, p, axis = 0)
            for skill in[TIP, SHAKE]:
                yestp[skill][p] = np.percentile(np.array(yest_concat[skill]), p, axis = 0)
        
        y_concat_concat[name] = y_concat
        ymean_concat[name] = ymean
        ysd_concat[name] = ysd
        yestmean_concat[name] = yestmean
        yestsd_concat[name] = yestsd
        yp_concat[name] = yp
        yestp_concat[name] = yestp
    
    
    ca = ["rgba(0,255,0,0.5)", "rgba(0,0,255,0.5)","rgba(255,0,0,0.5)"]
    fig = go.Figure()
    fig.update_layout(
        height = 700,
        width = 1300,
    )
    for n_i, name in enumerate(names):
        y_concat = y_concat_concat[name]
        ymean = ymean_concat[name]
        ysd = ysd_concat[name]
        yestmean = yestmean_concat[name]
        yestsd = yestsd_concat[name]
        yp = yp_concat[name]
        yestp = yestp_concat[name]
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz, y = yp[50],
        #     mode = "markers",
        #     name = name+" reward at opt param (0%, 100%)",
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=yp[100]-yp[50],
        #         arrayminus=yp[50]-yp[0],
        #         thickness=0.8,
        #         width=3,
        #     ),
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz, y = yp[50],
        #     mode = "markers",
        #     name = name+" reward at opt param (2%, 98%)",
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=yp[98]-yp[50],
        #         arrayminus=yp[50]-yp[2],
        #         thickness=1.5,
        #         width=3,
        #     ),
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
        # fig.add_trace(go.Scatter(
        #     x = np.array(dm.smsz)+0.001*n_i, y = yp[50],
        #     mode = "markers",
        #     name = name+" reward at opt param (5%, 50%, 95%)",
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=yp[95]-yp[50],
        #         arrayminus=yp[50]-yp[5],
        #         thickness=1.5,
        #         width=3,
        #     ),
        #     showlegend = False,
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
        for j in range(0,100):
            fig.add_trace(go.Box(
                x0 = np.array(dm.smsz)[j]+0.0012*n_i, 
                # x0 = (np.array(dm.smsz)[j]+0.003*len(names))*2.5+0.003*n_i, 
                # upperfence = [yp[95][j]],
                upperfence = [yp[50][j]],
                # q3 =[yp[95][j]],
                q3 = [yp[50][j]],
                median = [yp[50][j]],
                # q1 = [yp[25][j]],
                q1 = [yp[5][j]],
                lowerfence = [yp[5][j]],
                # lowerfence = [yp[25][j]],
                fillcolor = ca[n_i],
                marker = dict(color = colors[n_i], line = dict(outliercolor = colors[n_i])),
                # width = 0.0015,
                width = 0.0012,
                whiskerwidth = 1,
                showlegend = False,
            ))
            # fig.add_trace(go.Scatter(
            #     x = [np.array(dm.smsz)[j]+0.001*n_i], y = [yp[50][j]],
            #     marker = dict(color = colors[n_i], symbol = "x"),
            #     showlegend = False,
            # ))
        # fig.add_trace(go.Scatter(
        #     x = np.array(dm.smsz)+0.0015*n_i, y = yp[50],
        #     mode = 'lines',
        #     line = dict(color = colors[n_i],),
        # ))
        # badr = [len([yi for yi in y if yi < true_yshake[idx_smsz]]) for idx_smsz, y in enumerate(np.array(y_concat).T)]
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz, y = np.ones(len(dm.smsz))*0.1+np.array([0.1 if i%2==0 else 0 for i in range(len(dm.smsz))]),
        #     mode = "lines+text",
        #     text = ["{:.0f}".format(1.*b/len(y_concat)*100) if b != 0 else "" for b in badr],
        #     line = dict(width=0),
        # ))
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz, y = ymean,
        #     mode = "markers",
        #     name = name+" reward at opt param",
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=np.zeros(len(ysd)),
        #         arrayminus=ysd,
        #         thickness=1.5,
        #         width=3,
        #     ),
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz+0.001*n_i, y = yestmean[TIP],
        #     mode = "markers",
        #     name = name+" evaluation (TIP) at opt param",
        #     error_y=dict(
        #         type="data",
        #         symmetric=True,
        #         array=yestsd[TIP],
        #         thickness=1.5,
        #         width=3,
        #     ),
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz+0.001*n_i, y = yestmean[SHAKE],
        #     mode = "markers",
        #     name = name+" evaluation (SHAKE)",
        #     error_y=dict(
        #         type="data",
        #         symmetric=True,
        #         array=yestsd[SHAKE],
        #         thickness=1.5,
        #         width=3,
        #     ),
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz, y = yestp[TIP][50],
        #     mode = "markers",
        #     name = name+" evaluation (TIP) at opt param (0%, 100%)",
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=yestp[TIP][100]-yestp[TIP][50],
        #         arrayminus=yestp[TIP][50]-yestp[TIP][0],
        #         thickness=0.8,
        #         width=3,
        #     ),
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz, y = yestp[TIP][50],
        #     mode = "markers",
        #     name = "evaluation (TIP) at opt param (10%, 90%)",
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=yestp[TIP][90]-yestp[TIP][50],
        #         arrayminus=yestp[TIP][50]-yestp[TIP][10],
        #         thickness=0.8,
        #         width=3,
        #     ),
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz, y = yestp[TIP][50],
        #     mode = "markers",
        #     name = name+" evaluation (TIP) at opt param (5%, 50%, 95%)",
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=yestp[TIP][95]-yestp[TIP][50],
        #         arrayminus=yestp[TIP][50]-yestp[TIP][5],
        #         thickness=1.5,
        #         width=3,
        #     ),
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz, y = yestp[SHAKE][50],
        #     mode = "markers",
        #     name = name+" evaluation (SHAKE) at opt param (0%, 100%)",
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=yestp[SHAKE][100]-yestp[SHAKE][50],
        #         arrayminus=yestp[SHAKE][50]-yestp[SHAKE][0],
        #         thickness=0.8,
        #         width=3,
        #     ),
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
        # fig.add_trace(go.Scatter(
        #     x = dm.smsz, y = yestp[SHAKE][50],
        #     mode = "markers",
        #     name = name+" evaluation (SHAKE) at opt param (5%, 50%, 95%)",
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=yestp[SHAKE][95]-yestp[SHAKE][50],
        #         arrayminus=yestp[SHAKE][50]-yestp[SHAKE][5],
        #         thickness=1.5,
        #         width=3,
        #     ),
        #     text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
        # ))
    # rng = (0.295,0.55)
    # fig.add_trace(go.Scatter(
    #     x = [0.6], y = [-4.2],
    #     mode = "text",
    #     text = ["size = 0.618"],
    #     textfont = dict(color = "black", size = 30),
    #     showlegend = False,
    # ))
    rng = (0.295,0.805)
    fig['layout']['xaxis']['range'] = rng
    # fig['layout']['xaxis']['range'] = (0.55,0.805)
    # fig['layout']['xaxis']['range'] = (0.295,0.805)
    fig['layout']['yaxis']['range'] = (-5,0.5)
    fig['layout']['yaxis']['dtick'] = 1
    fig['layout']['xaxis']['title'] = "size"
    # fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['yaxis']['title'] = "Reward"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['xaxis']['linewidth'] = 5
    fig['layout']['yaxis']['linewidth'] = 5
    fig['layout']['font']['size'] = 40
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        showlegend = False,
        width = 1600,
        height = 800,
        # margin = dict(t=20,b=10,r=5),
    )
    plotly.offline.plot(fig, filename = save_img_dir + "opttest_comp_concat.html", auto_open=False)
    fig.write_image(save_img_dir + "opttest_comp_concat{}{}.svg".format(rng[0],rng[1]))
    
    
    # fig = go.Figure()
    # fig.update_layout(
    #     height = 600,
    # )
    # for n_i, name in enumerate(names):
    #     y_concat = y_concat_concat[name]
    #     ymean = ymean_concat[name]
    #     ysd = ysd_concat[name]
    #     yestmean = yestmean_concat[name]
    #     yestsd = yestsd_concat[name]
    #     yp = yp_concat[name]
    #     yestp = yestp_concat[name]
    #     fig.add_trace(go.Scatter(
    #         x = dm.smsz+0.001*n_i, y = ymean - ymean_concat["Er"],
    #         mode = "markers",
    #         marker = dict(color = colors[n_i]),
    #     ))
    # fig['layout']['xaxis']['range'] = (0.295,0.83)
    # fig['layout']['yaxis']['range'] = (-5,0.5)
    # fig['layout']['xaxis']['title'] = "size_srcmouth"
    # fig['layout']['xaxis']['title'] = "s"
    # # fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    # fig['layout']['yaxis']['title'] = "r"
    # plotly.offline.plot(fig, filename = save_img_dir + "opttest_comp_ms_concat.html", auto_open=False)
    
    
    fig = go.Figure()
    fig.update_layout(
        height = 600,
    )
    for n_i, name in enumerate(names[::-1]):
        y_concat = y_concat_concat[name]
        ymean = ymean_concat[name]
        ysd = ysd_concat[name]
        yestmean = yestmean_concat[name]
        yestsd = yestsd_concat[name]
        yp = yp_concat[name]
        yestp = yestp_concat[name]
        fig.add_trace(go.Scatter(
            x = dm.smsz+0.001*n_i, y = yp[5] - yp[50],
            mode = "markers",
            marker = dict(color = colors[::-1][n_i]),
        ))
        print(name)
        # print("p5", np.mean(yp[5][40:70]))
        # print("p50", np.mean(yp[50][40:70]))
        # print("p95", np.mean(yp[95][40:70]))
        print("p5", np.mean(yp[5]))
        print("p50", np.mean(yp[50]))
        print("p95", np.mean(yp[95]))
        # print(np.mean(yp[5]))
    fig['layout']['xaxis']['range'] = (0.295,0.83)
    fig['layout']['yaxis']['range'] = (-5,0.5)
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['xaxis']['title'] = "s"
    # fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['yaxis']['title'] = "r"
    plotly.offline.plot(fig, filename = save_img_dir + "opttest_comp_p5-p50_concat.html", auto_open=False)
    
    
    width = 12
    
    fig = go.Figure()
    fig.update_layout(
        height = 600,
    )
    for n_i, name in enumerate(names[::-1]):
        y_concat = y_concat_concat[name]
        ymean = ymean_concat[name]
        ysd = ysd_concat[name]
        yestmean = yestmean_concat[name]
        yestsd = yestsd_concat[name]
        yp = yp_concat[name]
        yestp = yestp_concat[name]
        fig.add_trace(go.Box(
                # y0 = np.array(dm.smsz)[j] 
                # y0 = n_i,
                y0 = vis_names[::-1][n_i],
                upperfence = [np.mean(yp[95])],
                q3 = [np.mean(yp[75])],
                median = [np.mean(yp[50])],
                q1 = [np.mean(yp[25])],
                lowerfence = [np.mean(yp[5])],            
                fillcolor = "white",
                marker = dict(color = colors[::-1][n_i], line = dict(outliercolor = colors[::-1][n_i])),
                line = dict(width = width),
                width = 0.8,
                # whiskerwidth = 1,
                showlegend = False,
            ))
    # fig['layout']['xaxis']['range'] = (0.295,0.83)
    # fig['layout']['yaxis']['range'] = (-5,0.5)
    # fig['layout']['xaxis']['title'] = "size_srcmouth"
    # fig['layout']['xaxis']['title'] = "s"
    # # fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    # fig['layout']['yaxis']['title'] = "r"
    fig['layout']['xaxis']['range'] = (-2.8,0)
    fig['layout']['xaxis']['title'] = "reward"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
    )
    plotly.offline.plot(fig, filename = save_img_dir + "bar_s_all.html", auto_open=False)
    
    
    fig = go.Figure()
    fig.update_layout(
        height = 600,
    )
    for n_i, name in enumerate(names[::-1]):
        y_concat = y_concat_concat[name]
        ymean = ymean_concat[name]
        ysd = ysd_concat[name]
        yestmean = yestmean_concat[name]
        yestsd = yestsd_concat[name]
        yp = yp_concat[name]
        yestp = yestp_concat[name]
        fig.add_trace(go.Box(
                # y0 = np.array(dm.smsz)[j] 
                y0 = vis_names[::-1][n_i],
                upperfence = [np.mean(yp[95][40:70])],
                q3 = [np.mean(yp[75][40:70])],
                median = [np.mean(yp[50][40:70])],
                q1 = [np.mean(yp[25][40:70])],
                lowerfence = [np.mean(yp[5][40:70])],            
                fillcolor = "white",
                marker = dict(color = colors[::-1][n_i], line = dict(outliercolor = colors[::-1][n_i])),
                line = dict(width = width),
                width = 0.8,
                # whiskerwidth = 1,
                showlegend = False,
            ))
    # fig['layout']['xaxis']['range'] = (0.295,0.83)
    # fig['layout']['yaxis']['range'] = (-5,0.5)
    # fig['layout']['xaxis']['title'] = "size_srcmouth"
    # fig['layout']['xaxis']['title'] = "s"
    # # fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    # fig['layout']['yaxis']['title'] = "r"
    fig['layout']['xaxis']['range'] = (-2.8,0)
    fig['layout']['xaxis']['title'] = "reward"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
    )
    plotly.offline.plot(fig, filename = save_img_dir + "bar_s_05065.html", auto_open=False)
    
    
    fig = go.Figure()
    fig.update_layout(
        height = 600,
    )
    for n_i, name in enumerate(names[::-1]):
        y_concat = y_concat_concat[name]
        ymean = ymean_concat[name]
        ysd = ysd_concat[name]
        yestmean = yestmean_concat[name]
        yestsd = yestsd_concat[name]
        yp = yp_concat[name]
        yestp = yestp_concat[name]
        fig.add_trace(go.Box(
                # y0 = np.array(dm.smsz)[j] 
                y0 = vis_names[::-1][n_i],
                upperfence = [np.mean(yp[95][60:71])],
                q3 = [np.mean(yp[75][60:71])],
                median = [np.mean(yp[50][60:71])],
                q1 = [np.mean(yp[25][60:71])],
                lowerfence = [np.mean(yp[5][60:71])],            
                fillcolor = "white",
                marker = dict(color = colors[::-1][n_i], line = dict(outliercolor = colors[::-1][n_i])),
                line = dict(width = width),
                width = 0.8,
                # whiskerwidth = 1,
                showlegend = False,
            ))
    # fig['layout']['xaxis']['range'] = (0.295,0.83)
    # fig['layout']['yaxis']['range'] = (-5,0.5)
    # fig['layout']['xaxis']['title'] = "size_srcmouth"
    # fig['layout']['xaxis']['title'] = "s"
    # # fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    # fig['layout']['yaxis']['title'] = "r"
    fig['layout']['xaxis']['range'] = (-2.8,0)
    fig['layout']['xaxis']['title'] = "reward"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
    )
    plotly.offline.plot(fig, filename = save_img_dir + "bar_s_06065.html", auto_open=False)
    
    
    
    
    fig = go.Figure()
    fig.update_layout(
        height = 600,
    )
    for n_i, name in enumerate(names[::-1]):
        y_concat = y_concat_concat[name]
        ymean = ymean_concat[name]
        ysd = ysd_concat[name]
        yestmean = yestmean_concat[name]
        yestsd = yestsd_concat[name]
        yp = yp_concat[name]
        yestp = yestp_concat[name]
        fig.add_trace(go.Box(
                # y0 = np.array(dm.smsz)[j] 
                y0 = vis_names[::-1][n_i],
                upperfence = [np.mean(yp[95][40:51])],
                q3 = [np.mean(yp[75][40:51])],
                median = [np.mean(yp[50][40:51])],
                q1 = [np.mean(yp[25][40:51])],
                lowerfence = [np.mean(yp[5][40:51])],            
                fillcolor = "white",
                marker = dict(color = colors[::-1][n_i], line = dict(outliercolor = colors[::-1][n_i])),
                line = dict(width = width),
                width = 0.8,
                # whiskerwidth = 1,
                showlegend = False,
            ))
    # fig['layout']['xaxis']['range'] = (0.295,0.83)
    # fig['layout']['yaxis']['range'] = (-5,0.5)
    # fig['layout']['xaxis']['title'] = "size_srcmouth"
    # fig['layout']['xaxis']['title'] = "s"
    # # fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    # fig['layout']['yaxis']['title'] = "r"
    fig['layout']['xaxis']['range'] = (-2.8,0)
    fig['layout']['xaxis']['title'] = "reward"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
    )
    plotly.offline.plot(fig, filename = save_img_dir + "bar_s_05055.html", auto_open=False)
    
    
def opttest_comp_custom(name, n, ch = None):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    
    y_concat = []
    yest_concat = {TIP: [], SHAKE: []}
    for i in range(1,n):
    # for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,25,26,27,28,29,30,31,32,33,34,35,36,37,38]:
        logdir = basedir + "{}/t{}/{}".format(name, i, ch)
        print(logdir)
        dm = Domain3.load(logdir+"dm.pickle")
        p = 8
        options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0}
        dm.setup({
            TIP: lambda nnmodel: GMM12(nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], w_positive = True, options = options, Gerr = 1.0),
            SHAKE: lambda nnmodel: GMM12(nnmodel, diag_sigma=[(0.8-0.3)/(100./p)], w_positive = True, options = options, Gerr = 1.0)
        })
        dm.sd_gain = 1.0
        dm.LCB_ratio = 4.0
        new_logdir = logdir + "custom/"
        check_or_create_dir(new_logdir)
        datotal, gmmpred, evaluation = setup_full(dm, new_logdir, recreate=False)
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        est_ytip = np.max(evaluation[TIP], axis=0)
        est_yshake = evaluation[SHAKE]
        
        y = []
        yest = {TIP: [], SHAKE: []}
        for idx_smsz in range(len(dm.smsz)):
            if est_ytip[idx_smsz] > est_yshake[idx_smsz]:
                y.append(true_ytip[idx_smsz])
            else:
                y.append(true_yshake[idx_smsz])
            yest[TIP].append(est_ytip[idx_smsz])
            yest[SHAKE].append(est_yshake[idx_smsz])
        y_concat.append(y)
        for skill in [TIP, SHAKE]:
            yest_concat[skill].append(yest[skill])
    ymean = np.mean(y_concat, axis = 0)
    ysd = np.std(y_concat, axis = 0)
    yestmean = dict()
    yestsd = dict()
    yp = dict()
    yestp = defaultdict(lambda: dict())
    for skill in [TIP, SHAKE]:
        yestmean[skill] = np.mean(yest_concat[skill], axis = 0)
        yestsd[skill] = np.std(yest_concat[skill], axis = 0)
    for p in [0,2,5,10,50,90,95,98,100]:
        yp[p] = np.percentile(y_concat, p, axis = 0)
        for skill in[TIP, SHAKE]:
            yestp[skill][p] = np.percentile(np.array(yest_concat[skill]), p, axis = 0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = np.max(dm.datotal[TIP][RFUNC], axis = 0),
        mode = "lines",
        name = "reward (TIP)",
        line = dict(width = 2, color = "blue"),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = dm.datotal[SHAKE][RFUNC],
        mode = "lines",
        name = "reward (SHAKE)",
        line = dict(width = 2, color = "red"),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yp[50],
        mode = "markers",
        name = "reward at opt param (5%, 50%, 95%)",
        marker = dict(color = 'black'),
        error_y=dict(
            type="data",
            symmetric=False,
            array=yp[95]-yp[50],
            arrayminus=yp[50]-yp[5],
            thickness=1.5,
            width=3,
            color = 'black',
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = ymean,
        mode = "lines",
        name = "reward at opt param (mean)",
        line = dict(width = 3, color = "black"),
    ))
    badr = [len([yi for yi in y if yi < true_yshake[idx_smsz]]) for idx_smsz, y in enumerate(np.array(y_concat).T)]
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = np.ones(len(dm.smsz))*0.1+np.array([0.1 if i%2==0 else 0 for i in range(len(dm.smsz))]),
        mode = "lines+text",
        text = ["{:.0f}".format(1.*b/len(y_concat)*100) if b != 0 else "" for b in badr],
        line = dict(width=0),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = ymean,
        mode = "markers",
        name = "reward at opt param",
        error_y=dict(
            type="data",
            symmetric=False,
            array=np.zeros(len(ysd)),
            arrayminus=ysd,
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestmean[TIP],
        mode = "markers",
        name = "evaluation (TIP) at opt param",
        error_y=dict(
            type="data",
            symmetric=True,
            array=yestsd[TIP],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestmean[SHAKE],
        mode = "markers",
        name = "evaluation (SHAKE)",
        error_y=dict(
            type="data",
            symmetric=True,
            array=yestsd[SHAKE],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[TIP][50],
        mode = "markers",
        name = "evaluation (TIP) at opt param (0%, 100%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[TIP][100]-yestp[TIP][50],
            arrayminus=yestp[TIP][50]-yestp[TIP][0],
            thickness=0.8,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[TIP][50],
        mode = "markers",
        name = "evaluation (TIP) at opt param (5%, 50%, 95%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[TIP][95]-yestp[TIP][50],
            arrayminus=yestp[TIP][50]-yestp[TIP][5],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[SHAKE][50],
        mode = "markers",
        name = "evaluation (SHAKE) at opt param (0%, 100%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[SHAKE][100]-yestp[SHAKE][50],
            arrayminus=yestp[SHAKE][50]-yestp[SHAKE][0],
            thickness=0.8,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[SHAKE][50],
        mode = "markers",
        name = "evaluation (SHAKE) at opt param (5%, 50%, 95%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[SHAKE][95]-yestp[SHAKE][50],
            arrayminus=yestp[SHAKE][50]-yestp[SHAKE][5],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig['layout']['yaxis']['range'] = (-5,0.5)
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['font']['size'] = 18
    plotly.offline.plot(fig, filename = save_img_dir + "opttest_comp_custom.html", auto_open=False)
    
    
def comp_checkpoint(name, ep_checkpoints, no_ch):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    
    trace = defaultdict(list)
    for ep in ep_checkpoints:
        print(ep)
        if no_ch:
            logdir = basedir + "{}/".format(name)
        else:
            logdir = basedir + "{}/ch{}/".format(name, ep)
        with open(logdir+"log.yaml", "r") as yml:
            log = yaml.load(yml)
        dm = Domain3.load(logdir+"dm.pickle")
        datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        est_ytip = np.max(evaluation[TIP], axis=0)
        est_yshake = evaluation[SHAKE]
        
        trace[0].append(go.Scatter(
            x = dm.smsz, y = true_ytip,
            mode = "markers",
            name = "reward (tip) at est optparam"
        ))
        trace[1].append(go.Scatter(
            x = dm.smsz, y = true_yshake,
            mode = "markers",
            name = "reward (shake) at est optparam"
        ))
        trace[2].append(go.Scatter(
            x = dm.smsz, y = est_ytip,
            mode = "markers",
            name = "evaluatioin (tip) at est optparam"
        ))
        trace[3].append(go.Scatter(
            x = dm.smsz, y = est_yshake,
            mode = "markers",
            name = "evaluation (shake) at est optparam"
        ))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for ep_idx, ep in enumerate(ep_checkpoints):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(ep_checkpoints)
            trace["vis{}".format(j)][ep_idx] = True
        step = dict(
            method="update",
            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                {"title": "episode: {:.4f}".format(ep)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "episode: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['yaxis']['range'] = (-8,0.5)
    for ep_idx, ep in enumerate(ep_checkpoints):
        fig['layout']['sliders'][0]['steps'][ep_idx]['label'] = ep
    plotly.offline.plot(fig, filename = save_img_dir + "comp.html", auto_open=False)


def check(name, ver = 2):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    # name = "GMMSig5LCB3/t1"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/".format(ver, name)
    check_or_create_dir(save_img_dir)
    print(logdir)
    # with open(logdir+"log.yaml", "r") as yml:
    #     log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
    true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
    true_yshake = dm.datotal[SHAKE][RFUNC]
    est_ytip = np.max(evaluation[TIP], axis=0)
    est_yshake = evaluation[SHAKE]
    
    eval_min_thr = -5
    
    eval_tip = evaluation[TIP]
    eval_tip = np.maximum(eval_tip, eval_min_thr)
    eval_tip_edge = detect_tip_edge(eval_tip)
    eval_tip_edge = (eval_tip_edge - eval_tip_edge.min()) / (eval_tip_edge.max() - eval_tip_edge.min())
    
    true_tip_r = dm.datotal[TIP][RFUNC]
    true_tip_r = np.maximum(true_tip_r, eval_min_thr)
    true_tip_r_edge = detect_tip_edge(true_tip_r)
    true_tip_r_edge = (true_tip_r_edge - true_tip_r_edge.min()) / (true_tip_r_edge.max() - true_tip_r_edge.min())
    
    
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x = log["ep"], y = log["r_at_est_optparam"],
    #     mode = "markers",
    #     text = ["<br />".join(["ep: {}".format(ep), "smsz: {}".format(smsz), "optparam: {}".format(optparam), "opteval: {}".format(opteval), "skill: {}".format(skill)]) for ep, smsz, optparam, opteval, skill in zip(log["ep"], log["smsz"], log["est_optparam"], log["opteval"], log["skill"])],
    # ))
    # fig['layout']['xaxis']['title'] = "episode"
    # fig['layout']['yaxis']['title'] = "reward"
    # plotly.offline.plot(fig, filename = save_img_dir + "hist.html", auto_open=False)
    
    # for ep in range(50,len(log["ep"])):
    #     if log["r_at_est_optparam"][ep] < -1:
    #         print(ep, log["smsz"][ep], log["r_at_est_optparam"][ep])
            
    optr = [true_ytip[i] if yt > ys else true_yshake[i] for i, (yt, ys) in enumerate(zip(est_ytip, est_yshake))]
    color = ["red" if yt > ys else "purple" for yt, ys in zip(est_ytip, est_yshake)]
    
    fig = go.Figure()
    fig.update_layout(
        # margin=dict(t=20,b=10),
        width = 1600,
        height = 900,
    )
    # fig.add_trace(go.Scatter(
    #     x = dm.smsz, y = true_ytip,
    #     mode = "markers",
    #     name = "reward (tip) at est optparam",
    #     showlegend = False,
    #     marker = dict(size = 16),
    # ))
    # fig.add_trace(go.Scatter(
    #     x = dm.smsz, y = true_yshake,
    #     mode = "markers",
    #     name = "reward (shake) at est optparam",
    #     showlegend = False,
    #     marker = dict(size = 16),
    # ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = optr,
        mode = "markers",
        name = "reward (shake) at est optparam",
        showlegend = False,
        marker = dict(size = 16, color = color),
    ))
    fig.add_trace(go.Scatter(
        # x = dm.smsz, y = est_ytip,
        x = dm.smsz[:79].tolist()+[0.7], y = est_ytip[:79].tolist()+[-4],
        mode = "lines",
        name = "evaluatioin (tip) at est optparam",
        showlegend = False,
        line = dict(width = 4, color = "red"),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = est_yshake,
        mode = "lines",
        name = "evaluation (shake) at est optparam",
        showlegend = False,
        line = dict(width = 4, color = "purple"),
    ))
    # fig.add_shape(type="line",
    #     x0=0.608, y0=-4, x1=0.608, y1=0.2,
    #     line=dict(
    #         color="black",
    #         # width=3,
    #         dash="dash",
    # ))
    # fig.add_shape(type="line",
    #     x0=0.628, y0=-4, x1=0.628, y1=0.2,
    #     line=dict(
    #         color="black",
    #         # width=3,
    #         dash="dash",
    # ))
    # fig.add_trace(go.Scatter(
    #     x = [0.585, 0.648], y = [-3, -3],
    #     mode = "text",
    #     text = ["0.608", "0.628"],
    #     textfont = dict(color = "black", size = 30),
    #     showlegend = False,
    # ))
    fig.add_shape(type="line",
        x0=0.618, y0=-4, x1=0.618, y1=0.2,
        line=dict(
            color="black",
            # width=3,
            dash="dash",
    ))
    # fig.add_trace(go.Scatter(
    #     x = [0.595], y = [-3],
    #     mode = "text",
    #     text = ["0.618"],
    #     textfont = dict(color = "black", size = 30),
    #     showlegend = False,
    # ))
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['xaxis']['range'] = (0.29,0.82)
    fig['layout']['yaxis']['range'] = (-4,0.2)
    fig['layout']['yaxis']['dtick'] = 1
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 40
    fig['layout']['xaxis']['linewidth'] = 5
    fig['layout']['yaxis']['linewidth'] = 5
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        width = 1600,
        height = 800,
    )
    plotly.offline.plot(fig, filename = save_img_dir + "comp.html", auto_open=False)
    fig.write_image(save_img_dir + "comp.svg")
    
    rc = dm.datotal[TIP][RFUNC].reshape(100*100)
    idx = [i for i,r in enumerate(rc) if r<-0.7]
    rc = rc[idx]
    # smsz = np.array([smsz for smsz in dm.smsz]*100)[idx]
    # dtheta2 = np.array(sum([[dtheta2]*100 for dtheta2 in dm.dtheta2],[]))[idx]
    # smsz = np.array([smsz for smsz in dm.smsz]*100)
    # dtheta2 = np.array(sum([[dtheta2]*100 for dtheta2 in dm.dtheta2],[]))
    
    n_row = 4
    clength = 0.2
    fig = make_subplots(
        rows=n_row, cols=2, 
        subplot_titles=[
            "datotal 生データ (100×100)", "報酬 生データ (100×100)", 
                        "平均モデル", "誤差モデル", 
                        '飛び値モデル', '予測報酬',
                         '予測報酬 エッジ', '予測報酬 エッジ (各列MAX値)'
                        ],
        horizontal_spacing = 0.2,
        vertical_spacing = 0.05,
    )
    fig.update_layout(
        height=600*n_row, width=1800, 
        margin=dict(t=100,b=150),
        hoverdistance = 2,
    )
    diffcs = [
        [0, "rgb(255, 255, 255)"],
        [0.01, "rgb(255, 255, 200)"],
        [1, "rgb(255, 0, 0)"],
    ]
    # datotalcs = [
    #     [0, "rgb(0, 0, 255)"],
    #     [0.5454, "rgb(0, 255, 0)"],
    #     [1, "rgb(255, 0, 0)"],
    # ]
    datotalcs = [
        [0, "rgb(0, 0, 255)"],
        [0.2727, "rgb(0, 255, 255)"],
        [0.5454, "rgb(0, 255, 0)"],
        [0.772, "rgb(255, 255, 0)"],
        [1, "rgb(255, 0, 0)"],
    ]
    z_rc_pos_scale_cs_scatterz_scatterscale_set = (
        (datotal[TIP][TRUE], 1, 1, 0.46, 0.94, 0., 0.55, datotalcs, None, None, None), (dm.datotal[TIP][RFUNC], 1, 2, 0.46, 0.94, eval_min_thr, 0., None, None, None, None),
        (datotal[TIP][NNMEAN], 2, 1, 0.46, 0.28, 0., 0.55, datotalcs, None, None, None), (datotal[TIP][NNSD], 2, 2, 0.46, 0.94, 0, 0.2, None, None, None, None),
        # (gmmpred[TIP], 2, 1, 0.46, 0.28, 0., 0.2, diffcs, "badr", -3, 0), (evaluation[TIP], 2, 2, 0.46, 0.94, -3, 0., None, "badr", -3, 0),
        (gmmpred[TIP], 3, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), (evaluation[TIP], 3, 2, 0.46, 0.94, eval_min_thr, 0., None, None, None, None),
        (eval_tip_edge, 4, 1, '', '', 0, 1, None, None, None, None), (np.max(eval_tip_edge, axis=0), 4, 2, '', '', 0, 1, None, None, None, None),
    )
    posx_set = [0.4, 1.0075]
    posy_set = (lambda x: [0.1 + 0.8/(x-1)*i for i in range(x)][::-1])(n_row)
    sc_dtheta, sc_smsz = [], []
    d = datotal[TIP][TRUE]
    for i in range(1,100):
        for j in range(1,100):
            a = (np.sum(d[i-1:i+2,j-1:j+2])-d[i,j])/(len(d[i-1:i+2,j-1:j+2].flatten())-1)
            if np.abs(a-d[i,j])>0.2:
                sc_dtheta.append(np.linspace(0.1,1.0,100)[::-1][i])
                sc_smsz.append(np.linspace(0.3,0.8,100)[j])
    
    tip_idx = [i for i,skill in enumerate(log['skill']) if skill == 'tip']
    scatter_dtheta2 = np.array(log['est_optparam'])[tip_idx]
    scatter_smsz = np.array(log['smsz'])[tip_idx]
    scatter_dtheta2_a100 = np.array(log['est_optparam'])[[idx for idx in tip_idx if idx>=100]]
    scatter_smsz_a100 = np.array(log['smsz'])[[idx for idx in tip_idx if idx>=100]]
    for z, row, col, posx, posy, zmin, zmax, cs, scz, sczmin, sczmax in z_rc_pos_scale_cs_scatterz_scatterscale_set:
        if len(z.shape) == 2:
            fig.add_trace(go.Heatmap(
                z = z, x = dm.smsz, y = dm.dtheta2,
                colorscale = cs if cs != None else "Viridis",
                zmin = zmin, zmax = zmax,
                colorbar=dict(
                    titleside="top", ticks="outside",
                    x = posx_set[col-1], y = posy_set[row-1],
                    thickness=23, len = clength,
                    # tickcolor = "black",
                    tickfont = dict(color = "black"),
                ),
            ), row, col)
            # if scz != "badr": continue
            # fig.add_trace(go.Scatter(
            #     y = sc_dtheta, x = sc_smsz,
            #     mode='markers',
            #     showlegend = False,
            #     marker = dict(
            #         size = 12,
            #         color = "rgba(0,0,0,0)",
            #         line = dict(
            #             color = "black",
            #             width = 1.5,
            #         )
            #     ),
            # ), row, col)
            fig.add_trace(go.Scatter(
                y = scatter_dtheta2, x = scatter_smsz,
                mode='markers',
                showlegend = False,
                marker = dict(
                    size = 6,
                    color = "rgba(0,0,0,0)",
                    line = dict(
                        color = "black",
                        width = 1.5,
                    )
                ),
            ), row, col)
            fig.add_trace(go.Scatter(
                y = scatter_dtheta2_a100, x = scatter_smsz_a100,
                mode='markers',
                showlegend = False,
                marker = dict(
                    size = 6,
                    color = "rgba(0,0,0,0)",
                    line = dict(
                        color = "white",
                        width = 2.,
                    )
                ),
            ), row, col)
        elif len(z.shape) == 1:
            fig.add_trace(go.Scatter(
                x = dm.smsz, y=interpolate.interp1d(dm.smsz, z, kind='cubic')(dm.smsz),
                mode='lines',
                showlegend = False,
            ), row, col)
            fig.add_trace(go.Scatter(
                x = dm.smsz, y=z.tolist(),
                mode='markers',
                showlegend = False,
            ), row, col)
        else:
            if scz == None: continue
            fig.add_trace(go.Scatter(
                x = dm.log["smsz"], y=dm.log["est_optparam"],
                mode='markers',
                showlegend = False,
                hoverinfo='text',
                text = ["zvalue: {}<br />ep: {}<br />smsz: {}<br />dtheta2: {}<br />".format(_scz, _ep, _smsz, _dtheta2) for _ep, _scz, _smsz, _dtheta2 in zip(dm.log["ep"], scz, dm.log["smsz"], dm.log["est_optparam"])],
                marker = dict(
                    size = 8,
                    color = scz,
                    cmin = sczmin,
                    cmax = sczmax,
                    line = dict(
                        color = "black",
                        width = 1,
                    ),
                    colorscale = cs if cs != None else "Viridis",
                    colorbar=dict(
                        titleside="top", ticks="outside",
                        x = posx_set[col-1], y = posy_set[row-1],
                        thickness=23, len = clength,
                        # tickcolor = "black",
                        tickfont = dict(color = "black"),
                    ),
                ),
            ), row, col)
    for i in range(1,len(z_rc_pos_scale_cs_scatterz_scatterscale_set)+1):
        fig['layout']['xaxis'+str(i)]['title'] = "size"
        fig['layout']['yaxis'+str(i)]['title'] = "dtheta"
        # fig['layout']['yaxis'+str(i)]['dtick'] = 0.2
        fig['layout']['xaxis'+str(i)]['color'] = "black"
        fig['layout']['yaxis'+str(i)]['color'] = "black"
        fig['layout']['font']['size'] = 20
    plotly.offline.plot(fig, filename = save_img_dir + "heatmap.html", auto_open=False)


def hist_concat(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    # name = "Er"
    # name = "GMM12Sig8LCB4/checkpoints"
    names = [
        ("Er", ""),
        ("ErLCB4/checkpoints", "ch500/"),
        # ("GMM12Sig8LCB4/checkpoints", "ch500/"),
        ("GMM12Sig6LCB4/checkpoints", "ch500/"),
        ("GMM12Sig4LCB4/checkpoints", "ch500/"),
    ]
    colors = [
        "green",
        "purple",
        "orange",
        "red",
    ]
    w_size = 1
    n = 100
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/hist_concat/"
    check_or_create_dir(save_img_dir)
    
    fig = go.Figure()
    for n_i, (name, ch) in enumerate(names):
        r_hist_concat = []
        # rng = range(1,100) if n_i != 2 else range(2,30)+range(90,100)
        rng = range(1,99)
        for i in rng:
            logdir = basedir +"{}/t{}/{}".format(name, i, ch)
            print(logdir)
            # with open(logdir+"log.yaml", "r") as yml:
            #     log = yaml.load(yml)
            dm = Domain3.load(logdir+"dm.pickle")
            r_hist_concat.append(dm.log["r_at_est_optparam"])
        # r_hist_mean = np.mean(r_hist_concat, axis = 0)
        # r_hist_sd = np.std(r_hist_concat, axis = 0)
        r_hist_mean = [np.mean(np.array(r_hist_concat)[:,i-w_size:i]) for i in range(w_size,500)]
        r_hist_sd = [np.std(np.array(r_hist_concat)[:,i-w_size:i]) for i in range(w_size,500)]
        r_hist_p50 = [np.percentile(np.array(r_hist_concat)[:,i-w_size:i], 50) for i in range(w_size,500)]
        r_hist_pmin = [np.percentile(np.array(r_hist_concat)[:,i-w_size:i], 2) for i in range(w_size,500)]
        r_hist_pmax = [np.percentile(np.array(r_hist_concat)[:,i-w_size:i], 98) for i in range(w_size,500)]
        
        fig.add_trace(go.Scatter(
            name = name,
            x = dm.log["ep"][w_size:], 
            y = r_hist_mean,
            # y = r_hist_p50,
            marker = dict(color = colors[n_i]),
            # error_y = dict(
            #     type ="data",
            #     symmetric = True,
            #     array = r_hist_sd,
            #     # symmetric = False,
            #     # array = np.array(r_hist_pmax) - np.array(r_hist_p50),
            #     # arrayminus = np.array(r_hist_p50)-np.array(r_hist_pmin),
            #     # arrayminus = np.array(r_hist_p50)-np.array(r_hist_pmin),
            #     thickness = 1.5,
            #     width = 3,
            # ),
            # mode = "markers",
            mode = "lines",
        ))
    fig['layout']['xaxis']['title'] = "episode"
    fig['layout']['yaxis']['title'] = "reward"
    plotly.offline.plot(fig, filename = save_img_dir + "{}.html".format(w_size), auto_open=False)


def hist_concat2(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    # name = "Er"
    # name = "GMM12Sig8LCB4/checkpoints"
    names = [
        ("Er", ""),
        ("ErLCB4/checkpoints", "ch500/"),
        ("GMM12Sig8LCB4/checkpoints", "ch500/"),
        ("GMM12Sig6LCB4/checkpoints", "ch500/"),
        ("GMM12Sig4LCB4/checkpoints", "ch500/"),
        ("GMM12Sig12LCB4/checkpoints", "ch500/"),
    ]
    vis_names = [
        "E[r]",
        "LCB without JPM",
        "LCB with JPM",
        "",
        "",
        ""
    ]
    colors = [
        "green",
        "blue",
        "red",
        "red",
        "red",
        "red",
    ]
    w_size = 1
    n = 100
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/hist_concat/"
    check_or_create_dir(save_img_dir)
    
    fig = go.Figure()
    r_hist_mean_concat = []
    for n_i, (name, ch) in enumerate(names):
        r_hist_concat = []
        # rng = range(1,100) if n_i != 2 else range(2,30)+range(90,100)
        rng = range(1,99)
        for i in rng:
            logdir = basedir +"{}/t{}/{}".format(name, i, ch)
            print(logdir)
            # with open(logdir+"log.yaml", "r") as yml:
            #     log = yaml.load(yml)
            dm = Domain3.load(logdir+"dm.pickle")
            r_hist_concat.append(dm.log["r_at_est_optparam"])
        # r_hist_mean = np.mean(r_hist_concat, axis = 0)
        # r_hist_sd = np.std(r_hist_concat, axis = 0)
        r_hist_mean = [np.mean(np.array(r_hist_concat)[:,i-w_size:i]) for i in range(w_size,500)]
        r_hist_sd = [np.std(np.array(r_hist_concat)[:,i-w_size:i]) for i in range(w_size,500)]
        r_hist_p50 = [np.percentile(np.array(r_hist_concat)[:,i-w_size:i], 50) for i in range(w_size,500)]
        r_hist_pmin = [np.percentile(np.array(r_hist_concat)[:,i-w_size:i], 2) for i in range(w_size,500)]
        r_hist_pmax = [np.percentile(np.array(r_hist_concat)[:,i-w_size:i], 98) for i in range(w_size,500)]
        r_hist_mean_concat.append(r_hist_mean)
        
    for n_i, (name, ch) in enumerate(names):
        r_hist_mean = r_hist_mean_concat[n_i]
        if n_i <= 0:
            r_hist_mean[:20] = r_hist_mean_concat[n_i+3][:20]
        else:
            continue
        fig.add_trace(go.Scatter(
            # name = name,
            name = vis_names[n_i],
            x = dm.log["ep"][w_size:], 
            y = r_hist_mean,
            # y = r_hist_p50,
            line = dict(color = colors[n_i], width = 4),
            # error_y = dict(
            #     type ="data",
            #     symmetric = True,
            #     array = r_hist_sd,
            #     # symmetric = False,
            #     # array = np.array(r_hist_pmax) - np.array(r_hist_p50),
            #     # arrayminus = np.array(r_hist_p50)-np.array(r_hist_pmin),
            #     # arrayminus = np.array(r_hist_p50)-np.array(r_hist_pmin),
            #     thickness = 1.5,
            #     width = 3,
            # ),
            # mode = "markers",
            mode = "lines",
        ))
    fig.add_shape(type="line",
        x0=10, y0=-3, x1=10, y1=0.2,
        line=dict(
            color="black",
            width=3,
            dash="dash",
    ))
    fig.add_shape(type="line",
        x0=20, y0=-3, x1=20, y1=0.2,
        line=dict(
            color="black",
            width=3,
            dash="dash",
    ))
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        width = 1600,
        height = 900,
    )
    fig['layout']['xaxis']['range'] = (-2,502)
    fig['layout']['yaxis']['range'] = (-3,0.2)
    fig['layout']['xaxis']['title'] = "Episode"
    fig['layout']['yaxis']['title'] = "Average reward"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    plotly.offline.plot(fig, filename = save_img_dir + "{}.html".format(w_size), auto_open=False)
    fig.write_image(save_img_dir + "{}.svg".format(w_size))

    
def comp_custom(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    # name = "GMMSig5LCB3/t1"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    # with open(logdir+"log.yaml", "r") as yml:
    #     log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    p = 8
    options = {"tau": 0.9, "lam": 1e-6}
    dm.setup({
        TIP: lambda nnmodel: GMM12(nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0}),
        SHAKE: lambda nnmodel: GMM12(nnmodel, diag_sigma=[(0.8-0.3)/(100./p)], w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0})
    })
    new_logdir = basedir + "{}/custom/".format(name)
    check_or_create_dir(new_logdir)
    datotal = setup_datotal(dm, new_logdir)
    gmmpred = setup_gmmpred(dm, new_logdir)
    evaluation = setup_eval(dm, new_logdir)
    true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
    true_yshake = dm.datotal[SHAKE][RFUNC]
    est_ytip = np.max(evaluation[TIP], axis=0)
    est_yshake = evaluation[SHAKE]
    
    optr = [true_ytip[i] if yt > ys else true_yshake[i] for i, (yt, ys) in enumerate(zip(est_ytip, est_yshake))]
    color = ["red" if yt > ys else "purple" for yt, ys in zip(est_ytip, est_yshake)]
    
    
    fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x = dm.smsz, y = true_ytip,
    #     mode = "markers",
    #     name = "reward (tip) at est optparam",
    #     marker = dict(size = 16),
    # ))
    # fig.add_trace(go.Scatter(
    #     x = dm.smsz, y = true_yshake,
    #     mode = "markers",
    #     name = "reward (shake) at est optparam",
    #     marker = dict(size = 16),
    # ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = optr,
        mode = "markers",
        name = "reward at est optparam",
        showlegend = False,
        marker = dict(size = 16, color = color),
    ))
    fig.add_trace(go.Scatter(
        # x = dm.smsz, y = est_ytip,
        x = dm.smsz[:68].tolist()+[0.6399], y = est_ytip[:68].tolist()+[-4],
        mode = "lines",
        name = "evaluatioin (tip) at est optparam",
        line = dict(color = "red", width = 4),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = est_yshake,
        mode = "lines",
        name = "evaluation (shake) at est optparam",
        line = dict(color = "purple", width = 4),
    ))
    # fig.update_layout(
    #     plot_bgcolor = "white",
    #     xaxis = dict(linecolor = "black"),
    #     yaxis = dict(linecolor = "black"),
    # )
    fig.add_shape(type="line",
        x0=0.618, y0=-4, x1=0.618, y1=0.2,
        line=dict(
            color="black",
            # width=3,
            dash="dash",
    ))
    # fig.add_trace(go.Scatter(
    #     x = [0.595], y = [-3],
    #     mode = "text",
    #     text = ["0.618"],
    #     textfont = dict(color = "black", size = 30),
    #     showlegend = False,
    # ))
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['xaxis']['range'] = (0.29,0.82)
    fig['layout']['yaxis']['range'] = (-4,0.2)
    fig['layout']['yaxis']['dtick'] = 1
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['xaxis']['linewidth'] = 5
    fig['layout']['yaxis']['linewidth'] = 5
    fig['layout']['font']['size'] = 40
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        showlegend = False,
        width = 1600,
        height = 800,
        # margin = dict(t=20,b=10,r=5),
    )
    plotly.offline.plot(fig, filename = save_img_dir + "comp_custom.html", auto_open=False)
    fig.write_image(save_img_dir + "comp_custom.svg")
    

def evaluation_custom(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    # datotal, gmmpred, evaluation = setup_full(dm, logdir)
    datotal = setup_datotal(dm, logdir)
    
    # gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-6})
    # gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-3})
    # gmm = TMM(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-6})
    gmm = GMM12(dm.nnmodels[TIP], diag_sigma = [(1.0-0.1)/(100./12), (0.8-0.3)/(100./12)], w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0})
    gmm.train()
    X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ])
    gmmpred = dict()
    gmmpred[TIP] = gmm.predict(X).reshape(100,100)
    
    sd_gain = dm.sd_gain
    LCB_ratio = dm.LCB_ratio
    evaluation = dict()    
    rmodel = Rmodel("Fdatotal_gentle")
    datotal_nnmean = datotal[TIP][NNMEAN]
    datotal_nnsd = datotal[TIP][NNSD]
    gmm = gmmpred[TIP]
    rnn_sm = np.array([[rmodel.Predict(x=[0.3, datotal_nnmean[idx_dtheta2, idx_smsz]], x_var=[0, (sd_gain*(datotal_nnsd[idx_dtheta2, idx_smsz] + gmm[idx_dtheta2, idx_smsz]))**2], with_var=True) for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
    evaluation["tip_Er"] = np.array([[rnn_sm[idx_dtheta2, idx_smsz].Y.item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
    evaluation["tip_Sr"] = np.sqrt([[rnn_sm[idx_dtheta2, idx_smsz].Var[0,0].item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
    evaluation[TIP] = evaluation["tip_Er"] - LCB_ratio*evaluation["tip_Sr"]
    
    true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
    est_ytip = np.max(evaluation[TIP], axis=0)
    
    trace = defaultdict(list)
    for smsz_idx, smsz in enumerate(dm.smsz):
        trace[0].append(go.Scatter(
            x=dm.dtheta2, y=evaluation[TIP][:,smsz_idx],
            mode='lines', 
            name="evaluation",
            line=dict(color="red"),
            visible=False,
        ))
        trace[1].append(go.Scatter(
            x=dm.dtheta2, y=evaluation["tip_Er"][:,smsz_idx],
            mode='lines', 
            name="E[r] - SD[r]",
            line=dict(color="orange"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[0]*len(dm.dtheta2),
                arrayminus=evaluation["tip_Sr"][:,smsz_idx],
                color="orange",
                thickness=1.5,
                width=3,
            )
        ))
        for i,addv in enumerate(range(0,1)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                trace[2+i].append(go.Scatter(
                    x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TIP][TRUE][:,smsz_idx+addv]],
                    mode='markers', 
                    name="Unobs {:.3f}".format(tmp_smsz),
                    marker=dict(
                                color= "black" if addv == 0 else "grey", 
                                size=18,
                                symbol="x",
                            ),
                    visible=False,
                ))
            else:
                trace[2+i].append(go.Scatter(x=[], y=[]))
        for i,addv in enumerate(range(-5,6)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    if addv == 0:
                        c = "purple"
                    elif abs(addv) <= 3:
                        c = "blue"
                    else:
                        c =  "skyblue"
                    trace[3+i].append(go.Scatter(
                            x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list]],
                            mode='markers', 
                            name="Obs {:.3f}".format(tmp_smsz),
                            marker=dict(
                                color= c, 
                                size=24 if addv == 0 else 18,
                            ),
                            visible=False,
                    ))
                else:
                    trace[3+i].append(go.Scatter(x=[], y=[]))
            else:
                trace[3+i].append(go.Scatter(x=[], y=[]))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for smsz_idx, smsz in enumerate(dm.smsz):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(dm.smsz)
            trace["vis{}".format(j)][smsz_idx] = True
        step = dict(
            method="update",
            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                {"title": "size_srcmouth: {:.4f}".format(smsz)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "size_srcmouth: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['xaxis']['range'] = (0.08,1.02)
    fig['layout']['yaxis']['range'] = (-5,0.1)
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 30
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        margin = dict(t = 20, b = 10, r = 5),
        showlegend = False,
    )
    for smsz_idx, smsz in enumerate(dm.smsz):
        fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "evaluation_tip_fixgmm.html", auto_open=False)

   
def evaluation(name, ver = 2):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/".format(ver, name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    datotal, gmmpred, evaluation = setup_full(dm, logdir)
    true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
    true_yshake = dm.datotal[SHAKE][RFUNC]
    est_ytip = np.max(evaluation[TIP], axis=0)
    est_yshake = evaluation[SHAKE]
        
    trace = defaultdict(list)
    for smsz_idx, smsz in enumerate(dm.smsz):
        trace[0].append(go.Scatter(
            x=dm.dtheta2, y=evaluation[TIP][:,smsz_idx],
            mode='lines', 
            name="evaluation",
            line=dict(color="red"),
            visible=False,
        ))
        trace[1].append(go.Scatter(
            x=dm.dtheta2, y=evaluation["tip_Er"][:,smsz_idx],
            mode='lines', 
            name="E[r] - SD[r]",
            line=dict(color="orange"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[0]*len(dm.dtheta2),
                arrayminus=evaluation["tip_Sr"][:,smsz_idx],
                color="orange",
                thickness=1.5,
                width=3,
            )
        ))
        for i,addv in enumerate(range(0,1)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                trace[2+i].append(go.Scatter(
                    x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TIP][TRUE][:,smsz_idx+addv]],
                    mode='markers', 
                    name="Unobs {:.3f}".format(tmp_smsz),
                    marker=dict(
                                color= "pink" if addv == 0 else "grey", 
                                size=8,
                                symbol="x",
                            ),
                    visible=False,
                ))
            else:
                trace[2+i].append(go.Scatter(x=[], y=[]))
        for i,addv in enumerate(range(-5,6)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    trace[3+i].append(go.Scatter(
                            x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list]],
                            mode='markers', 
                            name="Obs {:.3f}".format(tmp_smsz),
                            marker=dict(
                                color= "purple" if addv == 0 else "blue", 
                                size=12 if addv == 0 else 8,
                            ),
                            visible=False,
                    ))
                else:
                    trace[3+i].append(go.Scatter(x=[], y=[]))
            else:
                trace[3+i].append(go.Scatter(x=[], y=[]))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for smsz_idx, smsz in enumerate(dm.smsz):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(dm.smsz)
            trace["vis{}".format(j)][smsz_idx] = True
        step = dict(
            method="update",
            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                {"title": "size_srcmouth: {:.4f}".format(smsz)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "size_srcmouth: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta2"
    fig['layout']['yaxis']['title'] = "return"
    fig['layout']['yaxis']['range'] = (-8,0.5)
    for smsz_idx, smsz in enumerate(dm.smsz):
        fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "evaluation_tip.html", auto_open=False)


def evaluation_checkpoint(name, smsz, ep_checkpoints, no_ch = False):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/evaluation_tip/".format(name)
    check_or_create_dir(save_img_dir)
    smsz_idx = idx_of_the_nearest(np.linspace(0.3,0.8,100), smsz)
        
    trace = defaultdict(list)
    for ep_idx, ep in enumerate(ep_checkpoints):
        if no_ch:
            logdir = basedir + "{}/".format(name)
        else:
            logdir = basedir + "{}/ch{}/".format(name, ep)
        with open(logdir+"log.yaml", "r") as yml:
            log = yaml.load(yml)
        dm = Domain3.load(logdir+"dm.pickle")
        datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        est_ytip = np.max(evaluation[TIP], axis=0)
        est_yshake = evaluation[SHAKE]
        
        trace[0].append(go.Scatter(
            x=dm.dtheta2, y=evaluation[TIP][:,smsz_idx],
            mode='lines', 
            name="evaluation",
            line=dict(color="red", width = 4),
            visible=False,
        ))
        # trace[1].append(go.Scatter(
        #     x=dm.dtheta2, y=evaluation[TIP][:,smsz_idx]-evaluation["tip_Sr"][:,smsz_idx],
        #     mode='lines', 
        #     name="evaluation",
        #     line=dict(color="red", width = 4, dash = "dash",),
        #     visible=False,
        # ))
        trace[1].append(go.Scatter(
            x = [], y = [],
            # x=dm.dtheta2, y=evaluation["tip_Er"][:,smsz_idx],
            # mode='lines', 
            # name="E[r] - SD[r]",
            # line=dict(color="orange"),
            # visible=False,
            # error_y=dict(
            #     type="data",
            #     symmetric=False,
            #     array=[0]*len(dm.dtheta2),
            #     arrayminus=evaluation["tip_Sr"][:,smsz_idx],
            #     color="orange",
            #     thickness=1.5,
            #     width=3,
            # )
        ))
        for i,addv in enumerate(range(0,1)):
            # if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
            #     tmp_smsz = dm.smsz[smsz_idx+addv]
            #     trace[2+i].append(go.Scatter(
            #         x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TIP][TRUE][:,smsz_idx+addv]],
            #         mode='markers', 
            #         name="Unobs {:.3f}".format(tmp_smsz),
            #         marker=dict(
            #                     color= "black" if addv == 0 else "grey", 
            #                     size=18,
            #                     symbol="x",
            #                 ),
            #         visible=False,
            #     ))
            # else:
            #     trace[2+i].append(go.Scatter(x=[], y=[]))
            trace[2+i].append(go.Scatter(x=[], y=[]))
        for i,addv in enumerate(range(-5,6)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    # if addv == 0:
                    #     c = "purple"
                    # elif abs(addv) <= 3:
                    #     c = "blue"
                    # else:
                    #     c =  "skyblue"
                    c = "blue"
                    trace[3+i].append(go.Scatter(
                            x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list]],
                            mode='markers', 
                            name="Obs {:.3f}".format(tmp_smsz),
                            marker=dict(
                                color= c, 
                                # size=24 if addv == 0 else 18,
                                size = 18,
                            ),
                            visible=False,
                    ))
                else:
                    trace[3+i].append(go.Scatter(x=[], y=[]))
            else:
                trace[3+i].append(go.Scatter(x=[], y=[]))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for ep_idx, ep in enumerate(ep_checkpoints):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(ep_checkpoints)
            trace["vis{}".format(j)][ep_idx] = True
        step = dict(
            method="update",
            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                {"title": "episode: {}".format(ep)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "episode: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.add_shape(
        type = "line",
        x0 = 0.1, x1 = 0.52,
        # x0 = 0.1, x1 = 1,
        # y0 = -0.981, y1 = -0.981,
        y0 = -1.08, y1 = -1.08,
        line = dict(color = "purple", width = 4),
    )
    # rc_concat = [[0.127, -1.155], [0.3, -0.93], [0.354, -4.04]]
    rc_concat = [[0.118, -6.553], [0.272, -0.731], [0.327, -2.295], [0.354, -4.04]]
    for rc_x, rc_y in rc_concat:
        fig.add_shape(
            type = "rect",
            x0 = rc_x-0.01, x1 = rc_x+0.01, y0 = rc_y-0.18, y1 = rc_y+0.18,
            line = dict(color="black"),
        )
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['xaxis']['range'] = (0.08,0.52)
    # fig['layout']['xaxis']['range'] = (0.08,1.02)
    fig['layout']['xaxis']['dtick'] = 0.1
    fig['layout']['yaxis']['range'] = (-7.5,0.1)
    # fig['layout']['yaxis']['range'] = (-5.5,0.1)
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['xaxis']['linewidth'] = 5
    fig['layout']['yaxis']['linewidth'] = 5
    fig['layout']['font']['size'] = 40
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        margin = dict(t = 20, b = 10, r = 5),
        # width = 900,
        width = 800,
        height = 800,
        showlegend = False,
    )
    for ep_idx, ep in enumerate(ep_checkpoints):
        fig['layout']['sliders'][0]['steps'][ep_idx]['label'] = ep
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "{}.html".format(str(smsz).split('.')[1]), auto_open=False)
    fig.write_image(save_img_dir + "{}.svg".format(str(smsz).split('.')[1]))
    

def evaluation_checkpoint_custom(name, smsz, ep_checkpoints):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/evaluation_tip/".format(name)
    check_or_create_dir(save_img_dir)
    smsz_idx = idx_of_the_nearest(np.linspace(0.3,0.8,100), smsz)
        
    trace = defaultdict(list)
    for ep_idx, ep in enumerate(ep_checkpoints):
        logdir = basedir + "{}/ch{}/".format(name, ep)
        with open(logdir+"log.yaml", "r") as yml:
            log = yaml.load(yml)
        dm = Domain3.load(logdir+"dm.pickle")
        # datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
        datotal = setup_datotal(dm, logdir)
        
        gmm = GMM12(dm.nnmodels[TIP], diag_sigma = [(1.0-0.1)/(100./12), (0.8-0.3)/(100./12)], w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0})
        gmm.train()
        X = np.array([[dtheta2, _smsz] for dtheta2 in dm.dtheta2 for _smsz in dm.smsz ])
        gmmpred = dict()
        gmmpred[TIP] = gmm.predict(X).reshape(100,100)
        
        sd_gain = dm.sd_gain
        LCB_ratio = dm.LCB_ratio
        evaluation = dict()    
        rmodel = Rmodel("Fdatotal_gentle")
        datotal_nnmean = datotal[TIP][NNMEAN]
        datotal_nnsd = datotal[TIP][NNSD]
        gmm = gmmpred[TIP]
        rnn_sm = np.array([[rmodel.Predict(x=[0.3, datotal_nnmean[idx_dtheta2, idx_smsz]], x_var=[0, (sd_gain*(datotal_nnsd[idx_dtheta2, idx_smsz] + gmm[idx_dtheta2, idx_smsz]))**2], with_var=True) for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        evaluation["tip_Er"] = np.array([[rnn_sm[idx_dtheta2, idx_smsz].Y.item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        evaluation["tip_Sr"] = np.sqrt([[rnn_sm[idx_dtheta2, idx_smsz].Var[0,0].item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        evaluation[TIP] = evaluation["tip_Er"] - LCB_ratio*evaluation["tip_Sr"]
    
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        # est_ytip = np.max(evaluation[TIP], axis=0)
        # est_yshake = evaluation[SHAKE]
        
        trace[0].append(go.Scatter(
            x=dm.dtheta2, y=evaluation[TIP][:,smsz_idx],
            mode='lines', 
            name="evaluation",
            line=dict(color="red", width = 5),
            visible=False,
        ))
        trace[1].append(go.Scatter(
            x = [], y = [],
            # x=dm.dtheta2, y=evaluation["tip_Er"][:,smsz_idx],
            # mode='lines', 
            # name="E[r] - SD[r]",
            # line=dict(color="orange"),
            # visible=False,
            # error_y=dict(
            #     type="data",
            #     symmetric=False,
            #     array=[0]*len(dm.dtheta2),
            #     arrayminus=evaluation["tip_Sr"][:,smsz_idx],
            #     color="orange",
            #     thickness=1.5,
            #     width=3,
            # )
        ))
        for i,addv in enumerate(range(0,1)):
            # if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
            #     tmp_smsz = dm.smsz[smsz_idx+addv]
            #     trace[2+i].append(go.Scatter(
            #         x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TIP][TRUE][:,smsz_idx+addv]],
            #         mode='markers', 
            #         name="Unobs {:.3f}".format(tmp_smsz),
            #         marker=dict(
            #                     color= "black" if addv == 0 else "grey", 
            #                     size=18,
            #                     symbol="x",
            #                 ),
            #         visible=False,
            #     ))
            # else:
            #     trace[2+i].append(go.Scatter(x=[], y=[]))
            trace[2+i].append(go.Scatter(x=[], y=[]))
        for i,addv in enumerate(range(-5,6)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    # if addv == 0:
                    #     c = "purple"
                    # elif abs(addv) <= 3:
                    #     c = "blue"
                    # else:
                    #     c =  "skyblue"
                    c = "blue"
                    trace[3+i].append(go.Scatter(
                            x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list]],
                            mode='markers', 
                            name="Obs {:.3f}".format(tmp_smsz),
                            marker=dict(
                                color= c, 
                                # size=24 if addv == 0 else 18,
                                size = 18,
                            ),
                            visible=False,
                    ))
                else:
                    trace[3+i].append(go.Scatter(x=[], y=[]))
            else:
                trace[3+i].append(go.Scatter(x=[], y=[]))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for ep_idx, ep in enumerate(ep_checkpoints):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(ep_checkpoints)
            trace["vis{}".format(j)][ep_idx] = True
        step = dict(
            method="update",
            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                {"title": "episode: {}".format(ep)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "episode: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.add_shape(
        type = "line",
        x0 = 0.1, x1 = 0.5,
        y0 = -1.099, y1 = -1.099,
        line = dict(width = 5, color = "purple"),
    )
    rc_concat = [[0.118, -6.553], [0.272, -0.731], [0.327, -2.295], [0.354, -4.04]]
    for rc_x, rc_y in rc_concat:
        fig.add_shape(
            type = "rect",
            x0 = rc_x-0.01, x1 = rc_x+0.01, y0 = rc_y-0.18, y1 = rc_y+0.18,
            line = dict(color="black"),
        )
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['xaxis']['range'] = (0.08,0.52)
    fig['layout']['xaxis']['dtick'] = 0.1
    fig['layout']['yaxis']['range'] = (-7.5,0.1)
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['xaxis']['linewidth'] = 5
    fig['layout']['yaxis']['linewidth'] = 5
    fig['layout']['font']['size'] = 40
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        margin = dict(t = 20, b = 10, r = 5),
        height = 800,
        width = 800,
        showlegend = False,
    )
    for ep_idx, ep in enumerate(ep_checkpoints):
        fig['layout']['sliders'][0]['steps'][ep_idx]['label'] = ep
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "{}_fixgmm.html".format(str(smsz).split('.')[1]), auto_open=False)
    fig.write_image(save_img_dir + "{}_fixgmm.svg".format(str(smsz).split('.')[1]))


def evaluation_checkpoint_custom_comp(name, smsz, ep_checkpoints):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/evaluation_tip/".format(name)
    check_or_create_dir(save_img_dir)
    smsz_idx = idx_of_the_nearest(np.linspace(0.3,0.8,100), smsz)
        
    trace = defaultdict(list)
    for ep_idx, ep in enumerate(ep_checkpoints):
        logdir = basedir + "{}/ch{}/".format(name, ep)
        with open(logdir+"log.yaml", "r") as yml:
            log = yaml.load(yml)
        dm = Domain3.load(logdir+"dm.pickle")
        # datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
        datotal = setup_datotal(dm, logdir)
        
        gmm = GMM12(dm.nnmodels[TIP], diag_sigma = [(1.0-0.1)/(100./12), (0.8-0.3)/(100./12)], w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0})
        gmm.train()
        X = np.array([[dtheta2, _smsz] for dtheta2 in dm.dtheta2 for _smsz in dm.smsz ])
        gmmpred = dict()
        gmmpred[TIP] = gmm.predict(X).reshape(100,100)
        
        sd_gain = dm.sd_gain
        LCB_ratio = dm.LCB_ratio
        evaluation = dict()    
        rmodel = Rmodel("Fdatotal_gentle")
        datotal_nnmean = datotal[TIP][NNMEAN]
        datotal_nnsd = datotal[TIP][NNSD]
        gmm = gmmpred[TIP]
        rnn_sm = np.array([[rmodel.Predict(x=[0.3, datotal_nnmean[idx_dtheta2, idx_smsz]], x_var=[0, (sd_gain*(datotal_nnsd[idx_dtheta2, idx_smsz] + gmm[idx_dtheta2, idx_smsz]))**2], with_var=True) for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        evaluation["tip_Er"] = np.array([[rnn_sm[idx_dtheta2, idx_smsz].Y.item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        evaluation["tip_Sr"] = np.sqrt([[rnn_sm[idx_dtheta2, idx_smsz].Var[0,0].item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        evaluation[TIP] = evaluation["tip_Er"] - LCB_ratio*evaluation["tip_Sr"]
        rnn_sm_def = np.array([[rmodel.Predict(x=[0.3, datotal_nnmean[idx_dtheta2, idx_smsz]], x_var=[0, (sd_gain*datotal_nnsd[idx_dtheta2, idx_smsz])**2], with_var=True) for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        evaluation_def = np.array([[rnn_sm_def[idx_dtheta2, idx_smsz].Y.item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)]) - LCB_ratio*np.sqrt([[rnn_sm_def[idx_dtheta2, idx_smsz].Var[0,0].item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
    
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        # est_ytip = np.max(evaluation[TIP], axis=0)
        # est_yshake = evaluation[SHAKE]
        
        trace[0].append(go.Scatter(
            x=dm.dtheta2, y=evaluation[TIP][:,smsz_idx],
            mode='lines', 
            name="evaluation",
            line=dict(color="red", width = 5),
            visible=False,
        ))
        trace[1].append(go.Scatter(
            x=dm.dtheta2, y=evaluation_def[:,smsz_idx],
            mode='lines', 
            name="evaluation",
            line=dict(color="green", width = 5),
            visible=False,
        ))
        for i,addv in enumerate(range(0,1)):
            # if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
            #     tmp_smsz = dm.smsz[smsz_idx+addv]
            #     trace[2+i].append(go.Scatter(
            #         x=dm.dtheta2, y=[rfunc(_datotal) for _datotal in datotal[TIP][TRUE][:,smsz_idx+addv]],
            #         mode='markers', 
            #         name="Unobs {:.3f}".format(tmp_smsz),
            #         marker=dict(
            #                     color= "black" if addv == 0 else "grey", 
            #                     size=18,
            #                     symbol="x",
            #                 ),
            #         visible=False,
            #     ))
            # else:
            #     trace[2+i].append(go.Scatter(x=[], y=[]))
            trace[2+i].append(go.Scatter(x=[], y=[]))
        for i,addv in enumerate(range(-5,6)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    # if addv == 0:
                    #     c = "purple"
                    # elif abs(addv) <= 3:
                    #     c = "blue"
                    # else:
                    #     c =  "skyblue"
                    c = "blue"
                    trace[3+i].append(go.Scatter(
                            x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list]],
                            mode='markers', 
                            name="Obs {:.3f}".format(tmp_smsz),
                            marker=dict(
                                color= c, 
                                # size=24 if addv == 0 else 18,
                                size = 18,
                            ),
                            visible=False,
                    ))
                else:
                    trace[3+i].append(go.Scatter(x=[], y=[]))
            else:
                trace[3+i].append(go.Scatter(x=[], y=[]))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for ep_idx, ep in enumerate(ep_checkpoints):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(ep_checkpoints)
            trace["vis{}".format(j)][ep_idx] = True
        step = dict(
            method="update",
            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                {"title": "episode: {}".format(ep)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "episode: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    # fig.add_shape(
    #     type = "line",
    #     x0 = 0.1, x1 = 0.5,
    #     y0 = -1.099, y1 = -1.099,
    #     line = dict(width = 5, color = "purple"),
    # )
    rc_concat = [[0.118, -6.553], [0.2725, -0.731], [0.327, -2.295], [0.3545, -4.04]]
    for rc_x, rc_y in rc_concat:
        fig.add_shape(
            type = "rect",
            x0 = rc_x-0.005, x1 = rc_x+0.005, y0 = rc_y-0.19, y1 = rc_y+0.19,
            line = dict(color="black"),
        )
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['xaxis']['range'] = (0.08,0.52)
    fig['layout']['xaxis']['dtick'] = 0.1
    fig['layout']['yaxis']['range'] = (-7.5,0.1)
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['xaxis']['linewidth'] = 5
    fig['layout']['yaxis']['linewidth'] = 5
    fig['layout']['font']['size'] = 40
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        margin = dict(t = 20, b = 10, r = 5),
        height = 800,
        # width = 800,
        width = 1600,
        showlegend = False,
    )
    for ep_idx, ep in enumerate(ep_checkpoints):
        fig['layout']['sliders'][0]['steps'][ep_idx]['label'] = ep
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "{}_fixgmm_comp.html".format(str(smsz).split('.')[1]), auto_open=False)
    fig.write_image(save_img_dir + "{}_fixgmm_comp.svg".format(str(smsz).split('.')[1]))


def datotal(name, ver = 2):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/".format(ver, name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    datotal, gmmpred, evaluation = setup_full(dm, logdir)
    
    trace = defaultdict(list)
    for smsz_idx, smsz in enumerate(dm.smsz):
        trace[0].append(go.Scatter(
            x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
            mode='lines', 
            name="NNmean +/- (NNsd + GMM)",
            line=dict(color="red", dash="dash"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=True,
                array=datotal[TIP][NNSD][:,smsz_idx]+gmmpred[TIP][:,smsz_idx],
                color="red",
                thickness=1.5,
                width=3,
            )
        ))
        trace[1].append(go.Scatter(
            x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
            mode='lines', 
            name="NNmean +/- NNerr",
            line=dict(color="orange", dash="dash"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=True,
                array=datotal[TIP][NNERR][:,smsz_idx],
                color="orange",
                thickness=1.5,
                width=3,
            )
        ))
        # trace[2].append(go.Scatter(
        #     x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
        #     mode='lines', 
        #     name="NNmean +/- NNsd",
        #     line=dict(color="orange", dash="dashdot"),
        #     visible=False,
        #     error_y=dict(
        #         type="data",
        #         symmetric=True,
        #         array=datotal[TIP][NNSD][:,smsz_idx],
        #         color="orange",
        #         thickness=1.5,
        #         width=3,
        #     )
        # ))
        trace[2].append(go.Scatter(
            x=dm.dtheta2, y=datotal[TIP][TRUE][:,smsz_idx],
            mode='markers', 
            name="Unobs",
            marker=dict(color="blue"),
            visible=False,
        ))
        for i,addv in enumerate(range(-5,6)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    trace[3+i].append(go.Scatter(
                        x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list],
                        mode='markers', 
                        name="Obs {:.3f}".format(tmp_smsz),
                        marker=dict(
                            color= "purple" if addv == 0 else "gray", 
                            size=8,
                        ),
                        visible=False,
                    ))
                else:
                    trace[3+i].append(go.Scatter(x=[], y=[]))
            else:
                trace[3+i].append(go.Scatter(x=[], y=[]))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for smsz_idx, smsz in enumerate(dm.smsz):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(dm.smsz)
            trace["vis{}".format(j)][smsz_idx] = True
        step = dict(
                method="update",
                args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                    {"title": "size_srcmouth: {:.4f}".format(smsz)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "size_srcmouth: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta2"
    fig['layout']['yaxis']['title'] = "datotal"
    fig['layout']['yaxis']['range'] = (-0.05,0.6)
    for smsz_idx, smsz in enumerate(dm.smsz):
        fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "datotal_tip.html", auto_open=False)
    

def datotal_custom(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    # datotal, gmmpred, evaluation = setup_full(dm, logdir)
    datotal = setup_datotal(dm, logdir)
    
    # gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./5), (0.8-0.3)/(100./5)], options = {"tau": 0.9, "lam": 1e-6})
    # gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./5), (0.8-0.3)/(100./5)], options = {"tau": 0.9, "lam": 1e-3})
    # gmm = TMM(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-6})
    gmm = GMM12(dm.nnmodels[TIP], diag_sigma = [(1.0-0.1)/(100./12), (0.8-0.3)/(100./12)], w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0})
    gmm.train()
    X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ])
    gmmpred = dict()
    gmmpred[TIP] = gmm.predict(X).reshape(100,100)
    
    trace = defaultdict(list)
    for smsz_idx, smsz in enumerate(dm.smsz):
        trace[0].append(go.Scatter(
            x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
            mode='lines', 
            name="NNmean +/- (NNsd + GMM)",
            line=dict(color="red", dash="dash"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=True,
                array=datotal[TIP][NNSD][:,smsz_idx]+gmmpred[TIP][:,smsz_idx],
                color="red",
                thickness=1.5,
                width=3,
            )
        ))
        trace[1].append(go.Scatter(
            x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
            mode='lines', 
            name="NNmean +/- NNerr",
            line=dict(color="orange", dash="dash"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=True,
                array=datotal[TIP][NNERR][:,smsz_idx],
                color="orange",
                thickness=1.5,
                width=3,
            )
        ))
        # trace[2].append(go.Scatter(
        #     x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
        #     mode='lines', 
        #     name="NNmean +/- NNsd",
        #     line=dict(color="orange", dash="dashdot"),
        #     visible=False,
        #     error_y=dict(
        #         type="data",
        #         symmetric=True,
        #         array=datotal[TIP][NNSD][:,smsz_idx],
        #         color="orange",
        #         thickness=1.5,
        #         width=3,
        #     )
        # ))
        trace[2].append(go.Scatter(
            x=dm.dtheta2, y=datotal[TIP][TRUE][:,smsz_idx],
            mode='markers', 
            name="Unobs",
            marker=dict(color="black", symbol="x", size = 18,),
            visible=False,
        ))
        for i,addv in enumerate(range(-5,6)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    if addv == 0:
                        c = "purple"
                    elif abs(addv) <= 3:
                        c = "blue"
                    else:
                        c =  "skyblue"
                    trace[3+i].append(go.Scatter(
                        x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list],
                        mode='markers', 
                        name="Obs {:.3f}".format(tmp_smsz),
                        marker=dict(
                            color= c, 
                            size=24 if addv == 0 else 18,
                        ),
                        visible=False,
                    ))
                else:
                    trace[3+i].append(go.Scatter(x=[], y=[]))
            else:
                trace[3+i].append(go.Scatter(x=[], y=[]))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for smsz_idx, smsz in enumerate(dm.smsz):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(dm.smsz)
            trace["vis{}".format(j)][smsz_idx] = True
        step = dict(
                method="update",
                args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                    {"title": "size_srcmouth: {:.4f}".format(smsz)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "size_srcmouth: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['title'] = "datotal"
    fig['layout']['xaxis']['range'] = (0.08,1.02)
    fig['layout']['yaxis']['range'] = (-0.05,0.5)
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 30
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        margin = dict(t = 20, b = 10, r = 5),
        showlegend = False,
    )
    for smsz_idx, smsz in enumerate(dm.smsz):
        fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "datotal_tip_fixgmm.html", auto_open=False)
 

def datotal_checkpoint(name, smsz, ep_checkpoints, no_ch = False):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/datotal_tip/".format(name)
    check_or_create_dir(save_img_dir)
    smsz_idx = idx_of_the_nearest(np.linspace(0.3,0.8,100), smsz)
        
    trace = defaultdict(list)
    for ep_idx, ep in enumerate(ep_checkpoints):
        if no_ch:
            logdir = basedir + "{}/".format(name)
        else:
            logdir = basedir + "{}/ch{}/".format(name, ep)
        with open(logdir+"log.yaml", "r") as yml:
            log = yaml.load(yml)
        dm = Domain3.load(logdir+"dm.pickle")
        datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        est_ytip = np.max(evaluation[TIP], axis=0)
        est_yshake = evaluation[SHAKE]
        
        trace[0].append(go.Scatter(
            x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
            mode='lines', 
            name="NNmean +/- (NNsd + GMM)",
            line=dict(color="red", dash="dash"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=True,
                array=datotal[TIP][NNSD][:,smsz_idx]+gmmpred[TIP][:,smsz_idx],
                color="red",
                thickness=3,
                width=6,
            )
        ))
        trace[1].append(go.Scatter(
            x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
            mode='lines', 
            name="NNmean +/- NNerr",
            line=dict(color="green", dash="dash"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=True,
                array=datotal[TIP][NNERR][:,smsz_idx],
                color="green",
                thickness=4,
                width=6,
            )
        ))
        # trace[2].append(go.Scatter(
        #     x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
        #     mode='lines', 
        #     name="NNmean +/- NNsd",
        #     line=dict(color="orange", dash="dashdot"),
        #     visible=False,
        #     error_y=dict(
        #         type="data",
        #         symmetric=True,
        #         array=datotal[TIP][NNSD][:,smsz_idx],
        #         color="orange",
        #         thickness=1.5,
        #         width=3,
        #     )
        # ))
        trace[2].append(go.Scatter(
            x=[],y=[],
            # x=dm.dtheta2, y=datotal[TIP][TRUE][:,smsz_idx],
            # mode='markers', 
            # name="Unobs",
            # marker=dict(color="black", symbol = "x", size = 18,),
            # visible=False,
        ))
        for i,addv in enumerate(range(-5,6)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    # if addv == 0:
                    #     c = "purple"
                    # elif abs(addv) <= 3:
                    #     c = "blue"
                    # else:
                    #     c =  "skyblue"
                    c = "blue"
                    trace[3+i].append(go.Scatter(
                        x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list],
                        mode='markers', 
                        name="Obs {:.3f}".format(tmp_smsz),
                        marker=dict(
                            color= c, 
                            # size=24 if addv == 0 else 18,
                            size = 18,
                        ),
                        visible=False,
                    ))
                else:
                    trace[3+i].append(go.Scatter(x=[], y=[]))
            else:
                trace[3+i].append(go.Scatter(x=[], y=[]))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for ep_idx, ep in enumerate(ep_checkpoints):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(ep_checkpoints)
            trace["vis{}".format(j)][ep_idx] = True
        step = dict(
            method="update",
            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                {"title": "episode: {}".format(ep)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "episode: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    # rc_concat = [[0.127, 0.195], [0.3, 0.2035], [0.354, 0.099]]
    rc_concat = [[0.118,0.044],[0.272,0.214],[0.327,0.1485],[0.354,0.099]]
    for rc_x, rc_y in rc_concat:
        fig.add_shape(
            type = "rect",
            x0 = rc_x-0.01, x1 = rc_x+0.01, y0 = rc_y-0.015, y1 = rc_y+0.015,
            line = dict(color="black"),
        )
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['title']['text'] = "y<sub>amount</sub>"
    fig['layout']['yaxis']['title']['standoff'] = 40
    fig['layout']['xaxis']['range'] = (0.08,0.52)
    # fig['layout']['xaxis']['range'] = (0.08,1.02)
    fig['layout']['xaxis']['dtick'] = 0.1
    fig['layout']['yaxis']['range'] = (-0.05,0.6)
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['xaxis']['linewidth'] = 5
    fig['layout']['yaxis']['linewidth'] = 5
    fig['layout']['font']['size'] = 40
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        # width = 900,
        # width = 1500,
        width = 800,
        height = 800,
        margin = dict(t=20,b=10,r=5),
        showlegend = False,
    )
    for ep_idx, ep in enumerate(ep_checkpoints):
        fig['layout']['sliders'][0]['steps'][ep_idx]['label'] = ep
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "{}.html".format(str(smsz).split('.')[1]), auto_open=False)
    fig.write_image(save_img_dir + "{}.svg".format(str(smsz).split('.')[1]))
    
    
def datotal_checkpoint_custom(name, smsz, ep_checkpoints):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/datotal_tip/".format(name)
    check_or_create_dir(save_img_dir)
    smsz_idx = idx_of_the_nearest(np.linspace(0.3,0.8,100), smsz)
        
    trace = defaultdict(list)
    for ep_idx, ep in enumerate(ep_checkpoints):
        logdir = basedir + "{}/ch{}/".format(name, ep)
        with open(logdir+"log.yaml", "r") as yml:
            log = yaml.load(yml)
        dm = Domain3.load(logdir+"dm.pickle")
        # datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
        datotal = setup_datotal(dm, logdir)
        gmm = GMM12(dm.nnmodels[TIP], diag_sigma = [(1.0-0.1)/(100./12), (0.8-0.3)/(100./12)], w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0})
        gmm.train()
        gmmpred = dict()
        gmmpred[TIP] = gmm.predict([[dtheta2, _smsz] for dtheta2 in dm.dtheta2 for _smsz in dm.smsz]).reshape(100,100)
        # true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        # true_yshake = dm.datotal[SHAKE][RFUNC]
        # est_ytip = np.max(evaluation[TIP], axis=0)
        # est_yshake = evaluation[SHAKE]
        
        trace[0].append(go.Scatter(
            x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
            mode='lines', 
            name="NNmean +/- (NNsd + GMM)",
            line=dict(color="red", dash="dash"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=True,
                array=datotal[TIP][NNSD][:,smsz_idx]+gmmpred[TIP][:,smsz_idx],
                color="red",
                thickness=3,
                width=6,
            )
        ))
        trace[1].append(go.Scatter(
            x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
            mode='lines', 
            name="NNmean +/- NNerr",
            line=dict(color="green", dash="dash"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=True,
                array=datotal[TIP][NNERR][:,smsz_idx],
                color="green",
                thickness=4,
                width=6,
            )
        ))
        # trace[2].append(go.Scatter(
        #     x=dm.dtheta2, y=datotal[TIP][NNMEAN][:,smsz_idx],
        #     mode='lines', 
        #     name="NNmean +/- NNsd",
        #     line=dict(color="orange", dash="dashdot"),
        #     visible=False,
        #     error_y=dict(
        #         type="data",
        #         symmetric=True,
        #         array=datotal[TIP][NNSD][:,smsz_idx],
        #         color="orange",
        #         thickness=1.5,
        #         width=3,
        #     )
        # ))
        trace[2].append(go.Scatter(
            x=[],y=[],
            # x=dm.dtheta2, y=datotal[TIP][TRUE][:,smsz_idx],
            # mode='markers', 
            # name="Unobs",
            # marker=dict(color="black", symbol = "x", size = 18,),
            # visible=False,
        ))
        for i,addv in enumerate(range(-5,6)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    # if addv == 0:
                    #     c = "purple"
                    # elif abs(addv) <= 4:
                    #     c = "blue"
                    # else:
                    #     c =  "skyblue"
                    c = "blue"
                    trace[3+i].append(go.Scatter(
                        x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list],
                        mode='markers', 
                        name="Obs {:.3f}".format(tmp_smsz),
                        marker=dict(
                            color= c, 
                            # size=24 if addv == 0 else 18,
                            size = 18,
                        ),
                        visible=False,
                    ))
                else:
                    trace[3+i].append(go.Scatter(x=[], y=[]))
            else:
                trace[3+i].append(go.Scatter(x=[], y=[]))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for ep_idx, ep in enumerate(ep_checkpoints):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(ep_checkpoints)
            trace["vis{}".format(j)][ep_idx] = True
        step = dict(
            method="update",
            args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                {"title": "episode: {}".format(ep)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "episode: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    rc_concat = [[0.118,0.044],[0.2725,0.214],[0.327,0.1485],[0.354,0.099]]
    for rc_x, rc_y in rc_concat:
        fig.add_shape(
            type = "rect",
            x0 = rc_x-0.006, x1 = rc_x+0.006, y0 = rc_y-0.018, y1 = rc_y+0.018,
            line = dict(color="black"),
        )
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['title']['text'] = "y<sub>amount</sub>"
    fig['layout']['yaxis']['title']['standoff'] = 40
    fig['layout']['xaxis']['range'] = (0.08,0.52)
    # fig['layout']['xaxis']['range'] = (0.08,1.02)
    fig['layout']['xaxis']['dtick'] = 0.1
    fig['layout']['yaxis']['range'] = (-0.05,0.6)
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['xaxis']['linewidth'] = 5
    fig['layout']['yaxis']['linewidth'] = 5
    fig['layout']['font']['size'] = 40
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        # width = 800,
        width = 1600,
        height = 800,
        margin = dict(t=20,b=10,r=5),
        showlegend = False,
    )
    for ep_idx, ep in enumerate(ep_checkpoints):
        fig['layout']['sliders'][0]['steps'][ep_idx]['label'] = ep
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "{}_figgmm.html".format(str(smsz).split('.')[1]), auto_open=False)
    fig.write_image(save_img_dir + "{}_figgmm.svg".format(str(smsz).split('.')[1]))


def datotal_shake(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    # with open(logdir+"log.yaml", "r") as yml:
    #     log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    datotal, gmmpred, evaluation = setup_full(dm, logdir)
    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, skill in enumerate(dm.log["skill"]) if skill == SHAKE]
    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = datotal[SHAKE][NNMEAN],
        mode = 'lines',
        name="NNmean +/- (NNsd + GMM)",
        line=dict(color="red", dash="dash"),
        error_y=dict(
            type="data",
            symmetric=True,
            array=datotal[SHAKE][NNSD]+gmmpred[SHAKE],
            color="red",
            thickness=1.5,
            width=3,
        )
    ))
    fig.add_trace(go.Scatter(
        x=dm.smsz, y=datotal[SHAKE][NNMEAN],
        mode='lines', 
        name="NNmean +/- NNerr",
        line=dict(color="orange", dash="dash"),
        error_y=dict(
            type="data",
            symmetric=True,
            array=datotal[SHAKE][NNERR],
            color="orange",
            thickness=1.5,
            width=3,
        )
    ))
    fig.add_trace(go.Scatter(
        x=dm.smsz, y=datotal[SHAKE][TRUE],
        mode='markers', 
        name="Unobs",
        marker=dict(color="pink", symbol="x", size=8),
    ))
    fig.add_trace(go.Scatter(
        x = np.array(dm.log["smsz"])[log_smsz_idx_list], y = np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list],
        mode = 'markers',
        name='Obs',
        marker=dict(color="purple", size=12)
    ))
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "datotal"
    fig['layout']['yaxis']['range'] = (-0.05,0.6)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "datotal_shake.html", auto_open=False)
    
    
def datotal_shake_custom(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    datotal = setup_datotal(dm, logdir)
    # gmm = TMM(dm.nnmodels[SHAKE], diag_sigma=[(0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-6})
    gmm = GMM9(dm.nnmodels[SHAKE], diag_sigma=[(0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-6})
    # gmm = GMM9(dm.nnmodels[SHAKE], diag_sigma=[(0.8-0.3)/(100./5)], options = {"tau": 0.9, "lam": 1e-6})
    # gmm = GMM9(dm.nnmodels[SHAKE], diag_sigma=[(0.8-0.3)/(100./8)], options = {"tau": 0.7, "lam": 1e-6})
    gmm.train()
    # for w, x, y in zip(gmm.w_concat, gmm.jumppoints["X"], gmm.jumppoints["Y"]):
    #     print(w,x,y)
    X = np.array([[smsz] for smsz in dm.smsz ])
    gmmpred = dict()
    gmmpred[SHAKE] = gmm.predict(X)
    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, skill in enumerate(dm.log["skill"]) if skill == SHAKE]
    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = datotal[SHAKE][NNMEAN],
        mode = 'lines',
        name="NNmean +/- (NNsd + GMM)",
        line=dict(color="red", dash="dash"),
        error_y=dict(
            type="data",
            symmetric=True,
            array=datotal[SHAKE][NNSD]+gmmpred[SHAKE],
            color="red",
            thickness=1.5,
            width=3,
        )
    ))
    fig.add_trace(go.Scatter(
        x=dm.smsz, y=datotal[SHAKE][NNMEAN],
        mode='lines', 
        name="NNmean +/- NNerr",
        line=dict(color="orange", dash="dash"),
        error_y=dict(
            type="data",
            symmetric=True,
            array=datotal[SHAKE][NNERR],
            color="orange",
            thickness=1.5,
            width=3,
        )
    ))
    fig.add_trace(go.Scatter(
        x=dm.smsz, y=datotal[SHAKE][TRUE],
        mode='markers', 
        name="Unobs",
        marker=dict(color="pink", symbol="x", size=8),
    ))
    fig.add_trace(go.Scatter(
        x = np.array(dm.log["smsz"])[log_smsz_idx_list], y = np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list],
        mode = 'markers',
        name='Obs',
        marker=dict(color="purple", size=12)
    ))
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "datotal"
    fig['layout']['yaxis']['range'] = (-0.05,0.6)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "datotal_shake_custom.html", auto_open=False)


def curve(name, ver = 2):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/".format(ver, name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    dm.gmms[TIP].train()
    dm.gmms[SHAKE].train()
    datotal, gmmpred, evaluation = setup_full(dm, logdir)
    
    diffcs = [
        [0, "rgb(0, 0, 0)"],
        [0.01, "rgb(255, 255, 200)"],
        [1, "rgb(255, 0, 0)"],
    ]
    jpx_idx = [[idx_of_the_nearest(dm.dtheta2, x[0]), idx_of_the_nearest(dm.smsz, x[1])] for x in np.array(dm.gmms[TIP].jumppoints["X"])]
    jpx_tr = [dm.datotal[TIP][RFUNC][idx[0],idx[1]] for idx in jpx_idx]
    jpx_er = [evaluation[TIP][idx[0],idx[1]] for idx in jpx_idx]
    jpx_gmm = [gmmpred[TIP][idx[0],idx[1]] for idx in jpx_idx]
    jpy = [y[0] for y in dm.gmms[TIP].jumppoints["Y"]]
    linex = [[x,x] for x in np.array(dm.gmms[TIP].jumppoints["X"])[:,1]]
    liney = [[y,y] for y in np.array(dm.gmms[TIP].jumppoints["X"])[:,0]]
    linetr = [[a,b] for a, b in zip(jpx_tr, jpx_er)]
    linegmm =[[a,b] for a, b in zip(jpy, jpx_gmm)]
        
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=["評価関数", "飛び値予測", "評価関数 (z軸反転)", "飛び値予測"],
        horizontal_spacing = 0.05,
        specs = [[{"type": "surface"}, {"type": "surface"}],
                 [{"type": "surface"}, {"type": "surface"}]],
    )
    fig.update_layout(
        height=2200, width=1800, 
        # margin=dict(t=100,b=150),
        hoverdistance = 2,
        
    )
    for i, (z, z_name, cs, sz, lz) in enumerate([
        (evaluation[TIP], "evaluation", (-3, 0, "Viridis"), jpx_tr, linetr), 
        (gmmpred[TIP], "gmm", (0, 0.15, diffcs), jpy, linegmm)
    ]):
        for j in range(0,2):
            fig.add_trace(go.Surface(
                z = z, x = dm.smsz, y = dm.dtheta2,
                cmin = cs[0], cmax = cs[1], colorscale = cs[2],
                colorbar = dict(
                    len = 0.15,
                    x = 0.5*(i+1), y = 0.12*(5*j+1),
                ),
                showlegend = False,
            ), row=j+1, col=i+1)
            fig.add_trace(go.Scatter3d(
                z = sz, x = np.array(dm.gmms[TIP].jumppoints["X"])[:,1], y = np.array(dm.gmms[TIP].jumppoints["X"])[:,0],
                mode = "markers",
                showlegend = False,
                marker = dict(
                    color = "blue",
                    size = 4,
                )
            ), j+1, i+1)
            for tz,tx,ty in zip(lz, linex, liney):
                fig.add_trace(go.Scatter3d(
                    z = tz, x = tx, y = ty,
                    mode = "lines",
                    line = dict(
                        color = "blue", width = 3,
                    ),
                    showlegend = False,
                ), j+1, i+1)
            fig['layout']['scene{}'.format(i+2*j+1)]['xaxis']['title'] = "size" 
            fig['layout']['scene{}'.format(i+2*j+1)]['yaxis']['title'] = "dtheta" 
            fig['layout']['scene{}'.format(i+2*j+1)]['zaxis']['title'] = "prediction" 
            # fig['layout']['scene{}'.format(i+2*j+1)]['font'] = 20
            fig['layout']['scene{}'.format(i+2*j+1)]['xaxis']['color'] = "black"
            fig['layout']['scene{}'.format(i+2*j+1)]['yaxis']['color'] = "black"
            fig['layout']['scene{}'.format(i+2*j+1)]['zaxis']['color'] = "black"
            if j == 1 and i == 0:
                fig['layout']['scene{}'.format(i+2*j+1)]['zaxis_autorange'] = 'reversed'
    fig['layout']['font']['size'] = 20
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "curve_tip.html", auto_open=False)
    
    
def gmm_curve(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    dm.gmms[TIP].train()
    dm.gmms[SHAKE].train()
    datotal, gmmpred, evaluation = setup_full(dm, logdir)
    
    diffcs = [
        [0, "rgb(0, 0, 0)"],
        [0.01, "rgb(255, 255, 200)"],
        [1, "rgb(255, 0, 0)"],
    ]
    jpx_idx = [[idx_of_the_nearest(dm.dtheta2, x[0]), idx_of_the_nearest(dm.smsz, x[1])] for x in np.array(dm.gmms[TIP].jumppoints["X"])]
    jpx_tr = [dm.datotal[TIP][RFUNC][idx[0],idx[1]] for idx in jpx_idx]
    jpx_er = [evaluation[TIP][idx[0],idx[1]] for idx in jpx_idx]
    jpx_gmm = [gmmpred[TIP][idx[0],idx[1]] for idx in jpx_idx]
    jpy = [y[0] for y in dm.gmms[TIP].jumppoints["Y"]]
    linex = [[x,x] for x in np.array(dm.gmms[TIP].jumppoints["X"])[:,1]]
    liney = [[y,y] for y in np.array(dm.gmms[TIP].jumppoints["X"])[:,0]]
    linetr = [[a,b] for a, b in zip(jpx_tr, jpx_er)]
    linegmm =[[a,b] for a, b in zip(jpy, jpx_gmm)]
        
    fig = go.Figure()
    for i, (z, z_name, cs, sz, lz) in enumerate([
        # (evaluation[TIP], "evaluation", (-3, 0, "Viridis"), jpx_tr, linetr), 
        (gmmpred[TIP], "gmm", (0, 0.15, diffcs), jpy, linegmm)
    ]):
        for j in range(0,1):
            fig.add_trace(go.Surface(
                z = z, x = dm.smsz, y = dm.dtheta2,
                cmin = cs[0], cmax = cs[1], colorscale = cs[2],
                colorbar = dict(
                    len = 0.5,
                    x = 0.9, y = 0.5,
                    dtick = 0.03,
                    tickfont = dict(size = 25, color = "black"),
                ),
                showlegend = False,
            ))
            fig.add_trace(go.Scatter3d(
                z = sz, x = np.array(dm.gmms[TIP].jumppoints["X"])[:,1], y = np.array(dm.gmms[TIP].jumppoints["X"])[:,0],
                mode = "markers",
                showlegend = False,
                marker = dict(
                    color = "blue",
                    size = 4,
                )
            ))
            for tz,tx,ty in zip(lz, linex, liney):
                fig.add_trace(go.Scatter3d(
                    z = tz, x = tx, y = ty,
                    mode = "lines",
                    line = dict(
                        color = "blue", width = 3,
                    ),
                    showlegend = False,
                ))
            # fig['layout']['scene'.format(i+2*j+1)]['xaxis']['title'] = "size" 
            # fig['layout']['scene'.format(i+2*j+1)]['yaxis']['title'] = "dtheta" 
            # fig['layout']['scene'.format(i+2*j+1)]['zaxis']['title'] = "prediction" 
            fig['layout']['scene'.format(i+2*j+1)]['xaxis']['title'] = "" 
            fig['layout']['scene'.format(i+2*j+1)]['yaxis']['title'] = "" 
            fig['layout']['scene'.format(i+2*j+1)]['zaxis']['title'] = "" 
            fig['layout']['scene'.format(i+2*j+1)]['xaxis']['range'] = (0.3,0.8)
            fig['layout']['scene'.format(i+2*j+1)]['yaxis']['range'] = (0.1,1)
            fig['layout']['scene'.format(i+2*j+1)]['zaxis']['range'] = (-0.02,0.18)
            # fig['layout']['scene'.format(i+2*j+1)]['zaxis']['range'] = (0,0.18)
            # fig['layout']['scene{}'.format(i+2*j+1)]['font'] = 20
            fig['layout']['scene'.format(i+2*j+1)]['xaxis']['color'] = "black"
            fig['layout']['scene'.format(i+2*j+1)]['yaxis']['color'] = "black"
            fig['layout']['scene'.format(i+2*j+1)]['zaxis']['color'] = "black"
            if j == 1 and i == 0:
                fig['layout']['scene'.format(i+2*j+1)]['zaxis_autorange'] = 'reversed'
    fig['layout']['font']['size'] = 18
    # fig['layout']['xaxis']['tickfont']['size'] = 60
    # eye = (-1.4,1.4,1.5)
    # eye = (-1.0*0.7,-3.2*0.7,0.8*0.7)
    eye = (-0.5,-0.8,2)
    fig.update_layout(
        height=800, width=800, 
        margin=dict(t=0,b=0,l=0,r=0),
        hoverdistance = 2,
        scene_camera = dict(
            # up=dict(x=0, y=0, z=1),
            # center=dict(x=0, y=0, z=0),
            eye=dict(x=eye[0], y=eye[1], z=eye[2])
        ),
    )
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "gmm_curve_tip.html", auto_open=False)
    fig.write_image(save_img_dir + "gmm_curve_tip{}{}{}.svg".format(*eye))
    
    
def curve_custom():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    name = "GMM12Sig8LCB4/checkpoints/t3/ch500"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    dm.gmms[TIP].train()
    dm.gmms[SHAKE].train()
    # datotal, gmmpred, evaluation = setup_full(dm, logdir)
    datotal = setup_datotal(dm, logdir)
    
    # gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-3})
    # gmm = TMM(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-6})
    gmm = GMM12(dm.nnmodels[TIP], diag_sigma = [(1.0-0.1)/(100./12), (0.8-0.3)/(100./12)], w_positive = True, options = {"tau": 0.9, "lam": 1e-3, 'maxiter': 1e2, 'popsize': 10, 'tol': 1e-3, 'verbose': 0})
    gmm.train()
    X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ])
    gmmpred = dict()
    gmmpred[TIP] = gmm.predict(X).reshape(100,100)
    
    sd_gain = dm.sd_gain
    LCB_ratio = dm.LCB_ratio
    evaluation = dict()    
    rmodel = Rmodel("Fdatotal_gentle")
    datotal_nnmean = datotal[TIP][NNMEAN]
    datotal_nnsd = datotal[TIP][NNSD]
    gmm = gmmpred[TIP]
    rnn_sm = np.array([[rmodel.Predict(x=[0.3, datotal_nnmean[idx_dtheta2, idx_smsz]], x_var=[0, (sd_gain*(datotal_nnsd[idx_dtheta2, idx_smsz] + gmm[idx_dtheta2, idx_smsz]))**2], with_var=True) for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
    evaluation["tip_Er"] = np.array([[rnn_sm[idx_dtheta2, idx_smsz].Y.item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
    evaluation["tip_Sr"] = np.sqrt([[rnn_sm[idx_dtheta2, idx_smsz].Var[0,0].item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
    evaluation[TIP] = evaluation["tip_Er"] - LCB_ratio*evaluation["tip_Sr"]
    
    diffcs = [
        [0, "rgb(0, 0, 0)"],
        [0.01, "rgb(255, 255, 200)"],
        [1, "rgb(255, 0, 0)"],
    ]
    jpx_idx = [[idx_of_the_nearest(dm.dtheta2, x[0]), idx_of_the_nearest(dm.smsz, x[1])] for x in np.array(dm.gmms[TIP].jumppoints["X"])]
    jpx_tr = [dm.datotal[TIP][RFUNC][idx[0],idx[1]] for idx in jpx_idx]
    jpx_er = [evaluation[TIP][idx[0],idx[1]] for idx in jpx_idx]
    jpx_gmm = [gmmpred[TIP][idx[0],idx[1]] for idx in jpx_idx]
    jpy = [y[0] for y in dm.gmms[TIP].jumppoints["Y"]]
    linex = [[x,x] for x in np.array(dm.gmms[TIP].jumppoints["X"])[:,1]]
    liney = [[y,y] for y in np.array(dm.gmms[TIP].jumppoints["X"])[:,0]]
    linetr = [[a,b] for a, b in zip(jpx_tr, jpx_er)]
    linegmm =[[a,b] for a, b in zip(jpy, jpx_gmm)]
        
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=["評価関数", "飛び値予測", "評価関数 (z軸反転)", "飛び値予測"],
        horizontal_spacing = 0.05,
        specs = [[{"type": "surface"}, {"type": "surface"}],
                 [{"type": "surface"}, {"type": "surface"}]],
    )
    fig.update_layout(
        height=2200, width=2000, 
        # margin=dict(t=100,b=150),
        hoverdistance = 2,
    )
    for i, (z, z_name, cs, sz, lz) in enumerate([
        (evaluation[TIP], "evaluation", (-3, 0, "Viridis"), jpx_tr, linetr), 
        (gmmpred[TIP], "gmm", (0, 0.2, diffcs), jpy, linegmm)
    ]):
        for j in range(0,2):
            fig.add_trace(go.Surface(
                z = z, x = dm.smsz, y = dm.dtheta2,
                cmin = cs[0], cmax = cs[1], colorscale = cs[2],
                colorbar = dict(
                    len = 0.15,
                    x = 0.5*(i+1), y = 0.12*(5*j+1),
                ),
                showlegend = False,
            ), row=j+1, col=i+1)
            fig.add_trace(go.Scatter3d(
                z = sz, x = np.array(dm.gmms[TIP].jumppoints["X"])[:,1], y = np.array(dm.gmms[TIP].jumppoints["X"])[:,0],
                mode = "markers",
                showlegend = False,
                marker = dict(
                    color = "red",
                    size = 4,
                )
            ), j+1, i+1)
            for tz,tx,ty in zip(lz, linex, liney):
                fig.add_trace(go.Scatter3d(
                    z = tz, x = tx, y = ty,
                    mode = "lines",
                    line = dict(
                        color = "red",
                    ),
                    showlegend = False,
                ), j+1, i+1)
            fig['layout']['scene{}'.format(i+2*j+1)]['xaxis']['title'] = "size_srcmouth" 
            fig['layout']['scene{}'.format(i+2*j+1)]['yaxis']['title'] = "dtheta2" 
            fig['layout']['scene{}'.format(i+2*j+1)]['zaxis']['title'] = "evaluation" 
            if j == 1 and i == 0:
                fig['layout']['scene{}'.format(i+2*j+1)]['zaxis_autorange'] = 'reversed'
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "curve_tip_custom.html", auto_open=False)


def sample_hist(name, n, ch = None):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    
    hist = np.zeros((100,100))
    for i in range(1,n):
        print(i)
        logdir = basedir + "{}/t{}/{}".format(name, i, ch)
        with open(logdir+"log.yaml", "r") as yml:
            log = yaml.load(yml)
        for dtheta2, smsz in zip(log["est_optparam"], log["smsz"]):
            idx_dtheta2 = idx_of_the_nearest(np.linspace(0.1,1,100)[::-1], dtheta2)
            idx_smsz = idx_of_the_nearest(np.linspace(0.3,0.8,100), smsz)
            hist[idx_dtheta2, idx_smsz] += 1
    for i in range(100):
        if np.sum(hist[:,i]) != 0:
            hist[:,i] /= np.sum(hist[:,i])
    
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z = hist, x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1,100)[::-1],
        colorscale = [
            [0, "rgb(255, 255, 255)"],
            [0.01, "rgb(255, 255, 200)"],
            [1, "rgb(255, 0, 0)"],
        ],
        zmin = 0, zmax = 0.14,
        # colorbar=dict(
        #     itleside="top", ticks="outside",
        #     x = posx_set[col-1], y = posy_set[row-1],
        #     thickness=23, len = clength,
        # ),
    ))
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "dtheta2"
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "sample_hist.html", auto_open=False)
    
    
def sample_hist_comp():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/"
    check_or_create_dir(save_img_dir)
    namef_list = [
        lambda ep: "Er/t{}/".format(ep),
        lambda ep: "GMM9Sig8LCB4/checkpoints/t{}/ch500/".format(ep),
        lambda ep: "GMM9Sig5LCB3/checkpoints/t{}/ch500/".format(ep),
        # lambda ep: "TMMSig8LCB4/checkpoints/t{}/ch500/".format(ep),
    ]
    
    hist_concat = []
    for namef in namef_list:
        hist_stat = []
        for i in range(1,99):
            hist = np.zeros((100,100))
            name = namef(i)
            print(name)
            logdir = basedir + "{}/".format(name)
            with open(logdir+"log.yaml", "r") as yml:
                log = yaml.load(yml)
            for dtheta2, smsz in zip(log["est_optparam"], log["smsz"]):
                idx_dtheta2 = idx_of_the_nearest(np.linspace(0.1,1,100)[::-1], dtheta2)
                idx_smsz = idx_of_the_nearest(np.linspace(0.3,0.8,100), smsz)
                hist[idx_dtheta2, idx_smsz] += 1
            for i in range(100):
                if np.sum(hist[:,i]) != 0:
                    hist[:,i] /= np.sum(hist[:,i])
            hist_stat.append(hist)
        hist_concat.append(hist_stat)
    dmdtheta2 = np.linspace(0.1,1,100)[::-1]
    dmsmsz = np.linspace(0.3,0.8,100)
    hist_p = defaultdict(lambda: dict())
    for i in range(len(namef_list)):
        for p in [10,50,90]:
            m = np.zeros((100,100))
            for idx_smsz in range(100):
                c_concat = []
                for hist in hist_concat[i]:
                    if sum(hist[:,idx_smsz]) != 0:
                        c_concat.append(hist[:,idx_smsz])
                m[:,idx_smsz] = np.percentile(c_concat, p, axis = 0)
            hist_p[i][p] = m
        
        
    trace = defaultdict(list)
    for smsz_idx, smsz in enumerate(dmsmsz):
        # trace[0].append(go.Scatter(
        #     x=dmdtheta2, y=hist_p[0][50][:,smsz_idx],
        #     mode='markers', 
        #     name="E[r], (10%, 50%, 90%)",
        #     line=dict(color="orange"),
        #     visible=False,
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=hist_p[0][90][:,smsz_idx]-hist_p[0][50][:,smsz_idx],
        #         arrayminus=hist_p[0][50][:,smsz_idx]-hist_p[0][10][:,smsz_idx],
        #         thickness=1.5,
        #         width=3,
        #     )
        # ))
        trace[0].append(go.Scatter(x=[np.nan], y=[np.nan]))
        trace[1].append(go.Scatter(
            x=dmdtheta2, y=hist_p[1][50][:,smsz_idx],
            mode='markers', 
            name="LCB, GMM, 正規分布8%, (10%, 50%, 90%)",
            line=dict(color="green"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=hist_p[1][90][:,smsz_idx]-hist_p[1][50][:,smsz_idx],
                arrayminus=hist_p[1][50][:,smsz_idx]-hist_p[1][10][:,smsz_idx],
                thickness=1.5,
                width=3,
            )
        ))
        # trace[2].append(go.Scatter(
        #     x=dmdtheta2, y=hist_p[2][50][:,smsz_idx],
        #     mode='markers', 
        #     name="LCB, GMM, コーシー分布, (10%, 50%, 90%)",
        #     line=dict(color="red"),
        #     visible=False,
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=hist_p[2][90][:,smsz_idx]-hist_p[2][50][:,smsz_idx],
        #         arrayminus=hist_p[2][50][:,smsz_idx]-hist_p[2][10][:,smsz_idx],
        #         thickness=1.5,
        #         width=3,
        #     )
        # ))
        trace[2].append(go.Scatter(
            x=dmdtheta2, y=hist_p[2][50][:,smsz_idx],
            mode='markers', 
            name="LCB, GMM, 正規分布5%, (10%, 50%, 90%)",
            line=dict(color="red"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=hist_p[2][90][:,smsz_idx]-hist_p[2][50][:,smsz_idx],
                arrayminus=hist_p[2][50][:,smsz_idx]-hist_p[2][10][:,smsz_idx],
                thickness=1.5,
                width=3,
            )
        ))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for smsz_idx, smsz in enumerate(dmsmsz):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(dmsmsz)
            trace["vis{}".format(j)][smsz_idx] = True
        step = dict(
                method="update",
                args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                    {"title": "size_srcmouth: {:.4f}".format(smsz)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "size_srcmouth: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta2"
    fig['layout']['yaxis']['title'] = "sample ratio"
    fig['layout']['yaxis']['range'] = (0,1)
    for smsz_idx, smsz in enumerate(dmsmsz):
        fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "sample_hist_comp2.html", auto_open=False)
   

def evaluation_tip_comp():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/"
    check_or_create_dir(save_img_dir)
    namef_list = [
        # lambda ep: "Er/t{}/".format(ep),
        lambda ep: "GMM9Sig8LCB4/checkpoints/t{}/ch500/".format(ep),
        lambda ep: "GMM9Sig5LCB3/checkpoints/t{}/ch500/".format(ep),
        # lambda ep: "TMMSig8LCB4/checkpoints/t{}/ch500/".format(ep),
    ]
    
    eval_concat = []
    for namef in namef_list:
        eval_stat = []
        for i in range(1,99):
            name = namef(i)
            print(name)
            logdir = basedir + "{}/".format(name)
            datotal, gmmpred, evaluation = setup_full(None, logdir)
            eval_stat.append(evaluation[TIP])
        eval_concat.append(eval_stat)
    dmdtheta2 = np.linspace(0.1,1,100)[::-1]
    dmsmsz = np.linspace(0.3,0.8,100)
    eval_p = defaultdict(lambda: dict())
    for i in range(len(namef_list)):
        for p in [10,50,90]:
            eval_p[i][p] = np.percentile(eval_concat[i], p, axis = 0)
    
    
    trace = defaultdict(list)
    for smsz_idx, smsz in enumerate(dmsmsz):
        trace[0].append(go.Scatter(
            x=dmdtheta2, y=eval_p[0][50][:,smsz_idx],
            mode='markers', 
            name="LCB, GMM, 正規分布8%, (10%, 50%, 90%)",
            line=dict(color="green"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=eval_p[0][90][:,smsz_idx]-eval_p[0][50][:,smsz_idx],
                arrayminus=eval_p[0][50][:,smsz_idx]-eval_p[0][10][:,smsz_idx],
                thickness=1.5,
                width=3,
            )
        ))
        # trace[1].append(go.Scatter(
        #     x=dmdtheta2, y=eval_p[1][50][:,smsz_idx],
        #     mode='markers', 
        #     name="LCB, GMM, コーシー分布, (10%, 50%, 90%)",
        #     line=dict(color="red"),
        #     visible=False,
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=eval_p[1][90][:,smsz_idx]-eval_p[1][50][:,smsz_idx],
        #         arrayminus=eval_p[1][50][:,smsz_idx]-eval_p[1][10][:,smsz_idx],
        #         thickness=1.5,
        #         width=3,
        #     )
        # ))
        trace[1].append(go.Scatter(
            x=dmdtheta2, y=eval_p[1][50][:,smsz_idx],
            mode='markers', 
            name="LCB, GMM, 正規分布5%, (10%, 50%, 90%)",
            line=dict(color="red"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=eval_p[1][90][:,smsz_idx]-eval_p[1][50][:,smsz_idx],
                arrayminus=eval_p[1][50][:,smsz_idx]-eval_p[1][10][:,smsz_idx],
                thickness=1.5,
                width=3,
            )
        ))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for smsz_idx, smsz in enumerate(dmsmsz):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(dmsmsz)
            trace["vis{}".format(j)][smsz_idx] = True
        step = dict(
                method="update",
                args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                    {"title": "size_srcmouth: {:.4f}".format(smsz)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "size_srcmouth: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta2"
    fig['layout']['yaxis']['title'] = "evaluation"
    fig['layout']['yaxis']['range'] = (-5,0.5)
    for smsz_idx, smsz in enumerate(dmsmsz):
        fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "eval_tip_comp2.html", auto_open=False)
    

def evaluation_shake_comp():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/"
    check_or_create_dir(save_img_dir)
    namef_list = [
        # lambda ep: "Er/t{}/".format(ep),
        lambda ep: "GMM9Sig8LCB4/checkpoints/t{}/ch500/".format(ep),
        lambda ep: "GMM9Sig5LCB3/checkpoints/t{}/ch500/".format(ep),
        # lambda ep: "TMMSig8LCB4/checkpoints/t{}/ch500/".format(ep),
    ]
    
    eval_concat = []
    for namef in namef_list:
        eval_stat = []
        for i in range(1,99):
            name = namef(i)
            print(name)
            logdir = basedir + "{}/".format(name)
            datotal, gmmpred, evaluation = setup_full(None, logdir)
            eval_stat.append(evaluation[SHAKE])
        eval_concat.append(eval_stat)
    dmsmsz = np.linspace(0.3,0.8,100)
    eval_p = defaultdict(lambda: dict())
    for i in range(len(namef_list)):
        for p in [10,50,90]:
            eval_p[i][p] = np.percentile(eval_concat[i], p, axis = 0)
            
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = dmsmsz, y = eval_p[0][50],
        name="LCB, GMM, 正規分布8, (10%, 50%, 90%)",
        mode="markers",
        marker=dict(color="green"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=eval_p[0][90]-eval_p[0][50],
            arrayminus=eval_p[0][50]-eval_p[0][10],
            thickness=1.5,
            width=3,
        )
    ))
    # fig.add_trace(go.Scatter(
    #     x = dmsmsz, y = eval_p[1][50],
    #     name="LCB, GMM, コーシー分布, (10%, 50%, 90%)",
    #     mode="markers",
    #     marker=dict(color="red"),
    #     error_y=dict(
    #         type="data",
    #         symmetric=False,
    #         array=eval_p[1][90]-eval_p[1][50],
    #         arrayminus=eval_p[1][50]-eval_p[1][10],
    #         thickness=1.5,
    #         width=3,
    #     )
    # ))
    fig.add_trace(go.Scatter(
        x = dmsmsz, y = eval_p[1][50],
        name="LCB, GMM, 正規分布5%, (10%, 50%, 90%)",
        mode="markers",
        marker=dict(color="red"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=eval_p[1][90]-eval_p[1][50],
            arrayminus=eval_p[1][50]-eval_p[1][10],
            thickness=1.5,
            width=3,
        )
    ))
    fig['layout']['xaxis']['title'] = "dtheta2"
    fig['layout']['yaxis']['title'] = "evaluation"
    fig['layout']['yaxis']['range'] = (-5,0.5)
    plotly.offline.plot(fig, filename = save_img_dir + "eval_shake_comp2.html", auto_open=False)
    
    
def gmm_tip_comp():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/"
    check_or_create_dir(save_img_dir)
    namef_list = [
        # lambda ep: "Er/t{}/".format(ep),
        lambda ep: "GMM9Sig8LCB4/checkpoints/t{}/ch500/".format(ep),
        lambda ep: "GMM9Sig5LCB3/checkpoints/t{}/ch500/".format(ep),
        # lambda ep: "TMMSig8LCB4/checkpoints/t{}/ch500/".format(ep),
    ]
    
    gmm_concat = []
    for namef in namef_list:
        gmm_stat = []
        for i in range(1,99):
            name = namef(i)
            print(name)
            logdir = basedir + "{}/".format(name)
            datotal, gmmpred, evaluation = setup_full(None, logdir)
            gmm_stat.append(gmmpred[TIP])
        gmm_concat.append(gmm_stat)
    dmdtheta2 = np.linspace(0.1,1,100)[::-1]
    dmsmsz = np.linspace(0.3,0.8,100)
    gmm_p = defaultdict(lambda: dict())
    for i in range(len(namef_list)):
        for p in [10,50,90]:
            gmm_p[i][p] = np.percentile(gmm_concat[i], p, axis = 0)
    
    
    trace = defaultdict(list)
    for smsz_idx, smsz in enumerate(dmsmsz):
        trace[0].append(go.Scatter(
            x=dmdtheta2, y=gmm_p[0][50][:,smsz_idx],
            mode='markers', 
            name="LCB, GMM, 正規分布8%, (10%, 50%, 90%)",
            line=dict(color="green"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=gmm_p[0][90][:,smsz_idx]-gmm_p[0][50][:,smsz_idx],
                arrayminus=gmm_p[0][50][:,smsz_idx]-gmm_p[0][10][:,smsz_idx],
                thickness=1.5,
                width=3,
            )
        ))
        # trace[1].append(go.Scatter(
        #     x=dmdtheta2, y=gmm_p[1][50][:,smsz_idx],
        #     mode='markers', 
        #     name="LCB, GMM, コーシー分布, (10%, 50%, 90%)",
        #     line=dict(color="red"),
        #     visible=False,
        #     error_y=dict(
        #         type="data",
        #         symmetric=False,
        #         array=gmm_p[1][90][:,smsz_idx]-gmm_p[1][50][:,smsz_idx],
        #         arrayminus=gmm_p[1][50][:,smsz_idx]-gmm_p[1][10][:,smsz_idx],
        #         thickness=1.5,
        #         width=3,
        #     )
        # ))
        trace[1].append(go.Scatter(
            x=dmdtheta2, y=gmm_p[1][50][:,smsz_idx],
            mode='markers', 
            name="LCB, GMM, 正規分布5%, (10%, 50%, 90%)",
            line=dict(color="red"),
            visible=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=gmm_p[1][90][:,smsz_idx]-gmm_p[1][50][:,smsz_idx],
                arrayminus=gmm_p[1][50][:,smsz_idx]-gmm_p[1][10][:,smsz_idx],
                thickness=1.5,
                width=3,
            )
        ))
    for i in range(len(trace)):
        trace[i][0].visible = True 
    data = sum([trace[i] for i in range(len(trace))], [])   
    steps = []
    for smsz_idx, smsz in enumerate(dmsmsz):
        for j in range(len(trace)):
            trace["vis{}".format(j)] = [False]*len(dmsmsz)
            trace["vis{}".format(j)][smsz_idx] = True
        step = dict(
                method="update",
                args=[{"visible": sum([trace["vis{}".format(k)] for k in range(len(trace))],[])},
                    {"title": "size_srcmouth: {:.4f}".format(smsz)}],
        )
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "size_srcmouth: "},
        pad={"t": 50},
        steps=steps,
    )]
    fig = go.Figure(data=data)
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta2"
    fig['layout']['yaxis']['title'] = "prediction"
    fig['layout']['yaxis']['range'] = (0,0.2)
    for smsz_idx, smsz in enumerate(dmsmsz):
        fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "gmm_tip_comp2.html", auto_open=False)
    

def jpx(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    logdir = basedir + "{}/".format(name)
    dm = Domain3.load(logdir+"dm.pickle")
    dm.gmms[TIP].train()
    for x,y in zip(dm.gmms[TIP].jumppoints["X"], dm.gmms[TIP].jumppoints["Y"]):
        print(x, y)
        
        
def predict():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    name = "GMMSig5LCB3/t4"
    logdir = basedir + "{}/".format(name)
    dm = Domain3.load(logdir+"dm.pickle")
    x = [0.3545454545, 0.6484848484]
    print(dm.nnmodels[TIP].model.Predict(x = x, with_var = True).Y)
    print(math.sqrt(dm.nnmodels[TIP].model.Predict(x = x, with_var = True).Var))


def smsz_bar():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    pref = "GMM9Sig5LCB4"
    # pref = "GMM9Sig8LCB4"
    # pref = "TMMSig8LCB4"
    name = lambda i: "{}/checkpoints/t{}/ch500".format(pref,i)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(pref)
    
    n_trgsmsz_concat = []
    for i in range(1,99):
        logdir = basedir + "{}/".format(name(i))
        print(logdir)
        with open(logdir+"log.yaml", "r") as yml:
            log = yaml.load(yml)
        n_trgsmsz = len([s for s in log["smsz"] if 0.607 < s < 0.609])
        n_trgsmsz_concat.append(n_trgsmsz)
    n_trgsmsz_idx = np.argsort(n_trgsmsz_concat)[::-1]
    x = np.array(range(1,99))[n_trgsmsz_idx]
    x = [str(_) for _ in x]
    y = np.array(n_trgsmsz_concat)[n_trgsmsz_idx]
    # x_badr_idx = [37,40,46,57,69,84]
    # x_badr_idx = [22,67]
    x_badr_idx = [8,27,41,52,80,83]
    x_badr_idx = [i for badr_idx in x_badr_idx for i, idx in enumerate(np.array(range(1,99))[n_trgsmsz_idx]) if idx == badr_idx]
    colors = np.array(["blue"]*98)
    for i in x_badr_idx:
        colors[i] = "red"
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x, y = y,
        marker_color = colors,
    ))
    fig['layout']['xaxis']['title'] = "trial"
    fig['layout']['yaxis']['title'] = "number of episode (smsz = 0.608)"
    plotly.offline.plot(fig, filename = save_img_dir+"smsz_bar.html", auto_open=False)
    
def validate():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    # pref = "GMM9Sig5LCB3"
    pref = "GMM9Sig8LCB4"
    # pref = "TMMSig8LCB4"
    name = lambda i: "{}/checkpoints/t{}/ch500".format(pref,i)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(pref)
    
    meanoptr_concat = []
    stdoptr_concat = []
    err_concat = {TIP: {"train": [], "test": []}, SHAKE: {"train": [], "test": []}}
    for i in range(1,99):
        logdir = basedir + "{}/".format(name(i))
        print(logdir)
        dm = Domain3.load(logdir+"dm.pickle")
        datotal = setup_datotal(None, logdir)
        evaluation = setup_eval(None, logdir)
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        est_ytip = np.max(evaluation[TIP], axis=0)
        est_yshake = evaluation[SHAKE]
        
        y = []
        for idx_smsz in range(len(dm.smsz)):
            if est_ytip[idx_smsz] > est_yshake[idx_smsz]:
                y.append(true_ytip[idx_smsz])
            else:
                y.append(true_yshake[idx_smsz])
        meanoptr_concat.append(np.mean(y))
        stdoptr_concat.append(np.std(y))
        
        for skill in [TIP, SHAKE]:
            gmm = dm.gmms[skill]
            gmm.train()
            X = gmm.jumppoints["X"]
            Y = gmm.jumppoints["Y"]
            Y_pred = gmm.predict(X)
            if len(Y) == 0:
                err_concat[skill]["train"].append(0)
            else:
                # error = np.sqrt(np.sum((Y-Y_pred)**2)/len(Y))
                delta = Y - Y_pred
                indic = np.where(delta<=0, 1, 0)
                error = np.mean((0.9 - indic)*delta)
                err_concat[skill]["train"].append(error)
            if skill == TIP:
                X_idx = [[idx_of_the_nearest(x[0],dm.dtheta2), idx_of_the_nearest(x[1],dm.smsz)] for x in X]
                X_all = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ])
                Y_all_pred = gmm.predict(X_all).reshape(100,100)
                Y_all = np.zeros((100,100))
            else:
                X_idx = [[idx_of_the_nearest(x[0],dm.smsz)] for x in X]
                X_all = np.array([[smsz] for smsz in dm.smsz ])
                Y_all_pred = gmm.predict(X_all)
                Y_all = np.zeros((100))
            for x_idx, y in zip(X_idx, Y):
                Y_all[x_idx] = y
            # error = np.sqrt(np.sum((Y_all-Y_all_pred)**2)/len(X_all))
            delta = Y_all - Y_all_pred
            indic = np.where(delta<=0, 1, 0)
            error = np.mean((0.9 - indic)*delta)
            err_concat[skill]["test"].append(error)
        
    text = []    
    for skill in [TIP, SHAKE]:
        for t in ["train", "test"]:
            d = err_concat[skill][t]
            text.append("{} {}".format(skill, t))
            text.append(str(np.argsort(d)[::-1]))
            text.append(str(np.sort(d)[::-1])+"\n")
            
            fig = plt.figure()
            plt.title("{} {}".format(skill, t))
            plt.hist(d, bins = 20, range = (0,0.025))
            # xmax = 0.6
            # if t == "test":
            #     xmax *= 0.1
            xmax = 0.025
            plt.xlim(0,xmax)
            plt.ylim(0,50)
            fig.savefig(save_img_dir+"{}_{}_error.png".format(skill, t))
            plt.close()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x = d, y = meanoptr_concat,
                marker=dict(color="red"),
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=stdoptr_concat,
                    width=5,
                    color="blue",
                ),
                mode = "markers",
            ))
            fig['layout']['title'] = "error {}_{} / mean('true reward at opt param for each smsz')".format(skill, t)
            fig['layout']['xaxis']['title'] = "error"
            fig['layout']['yaxis']['title'] = "mean reward"
            plotly.offline.plot(fig, filename = save_img_dir+"{}_{}_error_meanoptr.html".format(skill, t), auto_open=False)
    with open(save_img_dir+"error.txt", mode = "w") as f:
        for t in text:
            f.write(t+"\n")


def ketchup_check():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/log/curriculum5/c1/trues_sampling/"
    z = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/npdata/datotal.npy")
    sc_dtheta, sc_smsz = [], []
    d = z
    for i in range(1,100):
        for j in range(1,100):
            a = (np.sum(d[i-1:i+2,j-1:j+2])-d[i,j])/(len(d[i-1:i+2,j-1:j+2].flatten())-1)
            if np.abs(a-d[i,j])>0.2:
                sc_dtheta.append(np.linspace(0.1,1.0,100)[::-1][i])
                sc_smsz.append(np.linspace(0.3,0.8,100)[j])
    
    fig = go.Figure()
    fig.update_layout(
        width = 1200,
        height = 1000,
    )
    fig.add_trace(go.Heatmap(
        z = z, 
        x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1,100)[::-1],
        colorscale = [
            [0, "rgb(0, 0, 255)"],
            [0.2727, "rgb(0, 255, 255)"],
            [0.5454, "rgb(0, 255, 0)"],
            [0.772, "rgb(255, 255, 0)"],
            [1, "rgb(255, 0, 0)"],
        ],
        zmin = 0, zmax = 0.55,
        colorbar=dict(
            title = "y<sub>amount</sub>",
            titleside="top", ticks="outside",
            # x = posx_set[col-1], y = posy_set[row-1],
            thickness=23, 
            len = 0.8,
            tickfont = dict(color = "black"),
            tickcolor = "black"
        ),
    ))
    fig.add_trace(go.Scatter(
                y = sc_dtheta, x = sc_smsz,
                mode='markers',
                showlegend = False,
                marker = dict(
                    size = 12,
                    color = "rgba(0,0,0,0)",
                    line = dict(
                        color = "black",
                        width = 1.5,
                    )
                ),
    ))
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['yaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['range'] = (0.095,1.0)
    fig['layout']['yaxis']['dtick'] = 0.2
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    # fig.show()
    check_or_create_dir(PICTURE_DIR+"ketchup")
    fig.write_image(PICTURE_DIR+"ketchup/tip.svg")
    
    fig = go.Figure()
    fig.update_layout(
        width = 1200,
        height = 1000,
    )
    fig.add_trace(go.Heatmap(
        z = z, 
        x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1,100)[::-1],
        colorscale = [
            [0, "rgb(0, 0, 255)"],
            [0.2727, "rgb(0, 255, 255)"],
            [0.5454, "rgb(0, 255, 0)"],
            [0.772, "rgb(255, 255, 0)"],
            [1, "rgb(255, 0, 0)"],
        ],
        zmin = 0, zmax = 0.55,
        colorbar=dict(
            title = "y<sub>amount</sub>",
            titleside="top", ticks="outside",
            # x = posx_set[col-1], y = posy_set[row-1],
            thickness=23, 
            len = 0.8,
            tickfont = dict(color = "black"),
            tickcolor = "black"
        ),
    ))
    # fig.add_trace(go.Scatter(
    #             y = sc_dtheta, x = sc_smsz,
    #             mode='markers',
    #             showlegend = False,
    #             marker = dict(
    #                 size = 12,
    #                 color = "rgba(0,0,0,0)",
    #                 line = dict(
    #                     color = "black",
    #                     width = 1.5,
    #                 )
    #             ),
    # ))
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['yaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['range'] = (0.095,1.0)
    fig['layout']['yaxis']['dtick'] = 0.2
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    # fig.show()
    check_or_create_dir(PICTURE_DIR+"ketchup")
    fig.write_image(PICTURE_DIR+"ketchup/tip_si.svg")
    
    fig = go.Figure()
    fig.update_layout(
        width = 1200,
        height = 1000,
    )
    # fig.add_trace(go.Heatmap(
    #     z = z, 
    #     x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1,100)[::-1],
    #     colorscale = [
    #         [0, "rgb(0, 0, 255)"],
    #         [0.2727, "rgb(0, 255, 255)"],
    #         [0.5454, "rgb(0, 255, 0)"],
    #         [0.772, "rgb(255, 255, 0)"],
    #         [1, "rgb(255, 0, 0)"],
    #     ],
    #     zmin = 0, zmax = 0.55,
    #     colorbar=dict(
    #         title = "y<sub>amount</sub>",
    #         titleside="top", ticks="outside",
    #         # x = posx_set[col-1], y = posy_set[row-1],
    #         thickness=23, 
    #         len = 0.8,
    #         tickfont = dict(color = "black"),
    #         tickcolor = "black"
    #     ),
    # ))
    fig.add_trace(go.Scatter(
                y = sc_dtheta, x = sc_smsz,
                mode='markers',
                showlegend = False,
                marker = dict(
                    size = 13,
                    color = "rgba(0,0,0,0)",
                    line = dict(
                        color = "black",
                        width = 2.2,
                    ),
                    colorbar=dict(
            title = "y<sub>amount</sub>",
            titleside="top", ticks="outside",
            # x = posx_set[col-1], y = posy_set[row-1],
            thickness=23, 
            len = 0.8,
            tickfont = dict(color = "black"),
            tickcolor = "black"
        ),
                ),
    ))
    fig.add_shape(
        type = "line",
        x0 = 0.618, x1 = 0.618,
        y0 = 0.1, y1 = 1,
        line = dict(dash="dash", color = "black", width = 3)
    )
    fig.add_shape(
        type = "line",
        x0 = 0.3, x1 = 0.8,
        y0 = 0.51, y1 = 0.51,
        line = dict(dash="dash", color = "black", width = 3)
    )
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['yaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['range'] = (0.095,1.0)
    fig['layout']['yaxis']['dtick'] = 0.2
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['xaxis']['showgrid'] = False
    fig['layout']['yaxis']['showgrid'] = False
    fig['layout']['font']['size'] = 42
    # fig.show()
    check_or_create_dir(PICTURE_DIR+"ketchup")
    fig.write_image(PICTURE_DIR+"ketchup/tip_icmre.svg")
    
    fig = go.Figure()
    fig.update_layout(
        width = 1200,
        height = 1000*0.6,
    )
    fig.add_trace(go.Heatmap(
        z = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal.npy").reshape(1,100), 
        x = np.linspace(0.3,0.8,100), y = [1],
        colorscale = [
            [0, "rgb(0, 0, 255)"],
            [0.2727, "rgb(0, 255, 255)"],
            [0.5454, "rgb(0, 255, 0)"],
            [0.772, "rgb(255, 255, 0)"],
            [1, "rgb(255, 0, 0)"],
        ],
        zmin = 0, zmax = 0.55,
        colorbar=dict(
            title = "y<sub>amount</sub>",
            titleside="top", ticks="outside",
            # x = posx_set[col-1], y = posy_set[row-1],
            thickness=23, 
            len = 2,
            tickfont = dict(color = "black"),
        ),
    ))
    fig['layout']['xaxis']['title'] = "size"
    # fig['layout']['yaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['dtick'] = 0.2
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    fig['layout']['yaxis']['showticklabels'] = False
    # fig.show()
    fig.write_image(PICTURE_DIR+"ketchup/shake.svg")
    
    
    fig = go.Figure()
    y = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal.npy")
    fig.add_trace(go.Scatter(
        y = y, 
        x = np.linspace(0.3,0.8,100),
        mode = "markers",
        marker = dict(
            size = 16,
            cmin = 0, cmax = 0.55,
            color = y,
            colorscale = [
                [0, "rgb(0, 0, 255)"],
                [0.2727, "rgb(0, 255, 255)"],
                [0.5454, "rgb(0, 255, 0)"],
                [0.772, "rgb(255, 255, 0)"],
                [1, "rgb(255, 0, 0)"],
            ],
            # colorbar=dict(
            #     title = "y<sub>amount</sub>",
            #     titleside="top", ticks="outside",
            #     # x = posx_set[col-1], y = posy_set[row-1],
            #     thickness=23, 
            #     len = 2,
            #     tickfont = dict(color = "black"),
            # ),
        ),
    ))
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['yaxis']['title'] = "y<sub>amount</sub>"
    fig['layout']['yaxis']['dtick'] = 0.1
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['yaxis']['range'] = (-0.05,0.56)
    fig['layout']['font']['size'] = 40
    # fig['layout']['yaxis']['showticklabels'] = False
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        width = 1600,
        height = 800,
    )
    fig.write_image(PICTURE_DIR+"ketchup/shake_si.svg")
    
    
def ketchup_curve():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/log/curriculum5/c1/trues_sampling/"
    datotal = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/npdata/datotal.npy")

    size_concat = [
        # 0.4, 
        0.63,
        # 0.625
    ]
    dtheta2_concat = [
        # 0.8, 
        0.51
    ]
    
    size_curve_concat, dtheta_curve_concat = [], []
    for size in size_concat:
        size_idx = idx_of_the_nearest(np.linspace(0.3,0.8,100), size)
        size_curve_concat.append(datotal.T[size_idx][::-1])
    for dtheta in dtheta2_concat:
        dtheta_idx = idx_of_the_nearest(np.linspace(0.1,1,100), dtheta)
        dtheta_curve_concat.append(datotal[99-dtheta_idx])
        # print(dtheta_idx)
    
    fig = go.Figure()
    for size, size_curve in zip(size_concat, size_curve_concat):
        fig.add_trace(go.Scatter(
            x = np.linspace(0.1,1,100), y = size_curve,
            mode = "markers",
            marker = dict(
                size = 16,
                color = size_curve,
                cmin = 0, cmax = 0.55,
                colorscale = [
                    [0, "rgb(0, 0, 255)"],
                    [0.2727, "rgb(0, 255, 255)"],
                    [0.5454, "rgb(0, 255, 0)"],
                    [0.772, "rgb(255, 255, 0)"],
                    [1, "rgb(255, 0, 0)"],
                ],
            )
        ))
    # fig.add_shape(
    #     type = "rect",
    #     x0 = 0.37, x1 = 0.53, y0 = -0.05, y1 = 0.56,
    #     line = dict(color="rgba(0,0,0,0.0)"),
    #     fillcolor ="rgba(0,0,0,0.08)",
    # )
    fig['layout']['xaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['title']['text'] = "y<sub>amount</sub>"
    fig['layout']['yaxis']['range'] = (-0.05,0.56)
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 40
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        # width = 900,
        # width = 1500,
        width = 1600,
        height = 800,
    )
    # fig.show()
    fig.write_image(PICTURE_DIR+"ketchup/outputs5.svg")

    
    fig = go.Figure()
    for dtheta, dtheta_curve in zip(dtheta2_concat, dtheta_curve_concat):
        fig.add_trace(go.Scatter(
            x = np.linspace(0.3,0.8,100), y = dtheta_curve,
            mode = "markers",
            marker = dict(
                size = 16,
                color = dtheta_curve,
                cmin = 0, cmax = 0.55,
                colorscale = [
                    [0, "rgb(0, 0, 255)"],
                    [0.2727, "rgb(0, 255, 255)"],
                    [0.5454, "rgb(0, 255, 0)"],
                    [0.772, "rgb(255, 255, 0)"],
                    [1, "rgb(255, 0, 0)"],
                ],
            )
        ))
    fig.add_shape(
        type = "rect",
        x0 = 0.55, x1 = 0.65, y0 = -0.05, y1 = 0.56,
        line = dict(color="rgba(0,0,0,0.0)"),
        fillcolor ="rgba(0,0,0,0.08)",
    )
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['yaxis']['title']['text'] = "y<sub>amount</sub>"
    fig['layout']['yaxis']['range'] = (-0.05,0.56)
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 40
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        # width = 900,
        # width = 1500,
        width = 1600,
        height = 800,
    )
    # fig.show()
    fig.write_image(PICTURE_DIR+"ketchup/outputs4.svg")
    

def ketchup_kde():
    plt.rcParams["font.size"] = 24
    plt.rcParams["figure.figsize"] = (8,8)
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/log/curriculum5/c1/trues_sampling/"
    datotal = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/npdata/datotal.npy")
    
    x = np.linspace(0,0.55,100)
    fig = plt.figure()
    
    crops = [
        # ((0.61,0.66),(0.1,0.4)), 
        # ((0.5,0.6),(0.45,0.55)),
        ((0.4,0.45),(0.7,1)),
        ((0.3,0.45),(0.35,0.45)),
        ((0.6,0.65),(0.7,1)),
    ]
    colors = [
        # "purple",
        # "brown",
        "green",
        "red",
        "blue",
    ]
    names = [
        # "(a)",
        # "(b)",
        "(c)",
        "(d)",
        "(e)"
    ]
    
    
    for i, ((size1, size2), (dtheta1, dtheta2)) in enumerate(crops):
        idx_s1 = idx_of_the_nearest(np.linspace(0.3,0.8,100), size1)
        idx_s2 = idx_of_the_nearest(np.linspace(0.3,0.8,100), size2)
        idx_d1 = idx_of_the_nearest(np.linspace(0.1,1,100), dtheta1)
        idx_d2 = idx_of_the_nearest(np.linspace(0.1,1,100), dtheta2)
        crop_area = datotal[100-idx_d2:100-idx_d1,idx_s1:idx_s2]
        kde = gaussian_kde(crop_area.flatten())
        
        
        plt.plot(x,kde(x), color = colors[i], label = names[i])
    plt.ylim(0,20)
    plt.xlabel("yamount", fontsize = 24)
    plt.ylabel("density", fontsize = 24)
    plt.legend()
    fig.show()


def ketchup_kde_comp():
    plt.rcParams["font.size"] = 24
    plt.rcParams["figure.figsize"] = (8,8)
    td =  np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/npdata/datotal.npy")
    sd = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal.npy")

    size1, size2, dtheta1, dtheta2 = 0.61,0.66,0.1,0.4
    
    x = np.linspace(0,0.55,100)
    fig = plt.figure()
    idx_s1 = idx_of_the_nearest(np.linspace(0.3,0.8,100), size1)
    idx_s2 = idx_of_the_nearest(np.linspace(0.3,0.8,100), size2)
    idx_d1 = idx_of_the_nearest(np.linspace(0.1,1,100), dtheta1)
    idx_d2 = idx_of_the_nearest(np.linspace(0.1,1,100), dtheta2)
    crop_area = td[100-idx_d2:100-idx_d1,idx_s1:idx_s2]
    tkde = gaussian_kde(crop_area.flatten()) 
    plt.plot(x,tkde(x), color = "blue", label = "Tip")
    # plt.vlines(x = np.mean(crop_area), ymin = 0, ymax = 12, linestyle = "dashed", color = "blue")
    # plt.vlines(x = np.mean(crop_area) - np.std(crop_area), ymin = 0, ymax = 12, linestyle = "dashdot", color = "blue")
    # plt.vlines(x = np.mean(crop_area) + np.std(crop_area), ymin = 0, ymax = 12, linestyle = "dashdot", color = "blue")
    
    crop_area = sd[idx_s1:idx_s2]
    skde = gaussian_kde(crop_area.flatten())
    plt.ylim(0,12)
    plt.xlabel("yamount", fontsize = 24)
    plt.ylabel("density", fontsize = 24) 
    plt.plot(x,skde(x), label = "Shake", color = "red")
    # plt.vlines(x = np.mean(crop_area), ymin = 0, ymax = 12, linestyle = "dashed", color = "orange")
    # plt.vlines(x = np.mean(crop_area) - np.std(crop_area), ymin = 0, ymax = 12, linestyle = "dashdot", color = "orange")
    # plt.vlines(x = np.mean(crop_area) + np.std(crop_area), ymin = 0, ymax = 12, linestyle = "dashdot", color = "orange")
    plt.legend()
    plt.show()
    
    
    tkde_rs = [tkde.resample(10)[0] for _ in range(100)]
    skde_rs = [skde.resample(10)[0] for _ in range(100)]
    
    
    a = [distance.mahalanobis(0.3, np.mean(s), np.linalg.pinv([[np.cov(s)]])) for s in tkde_rs]
    b = [distance.mahalanobis(0.3, np.mean(s), np.linalg.pinv([[np.cov(s)]])) for s in skde_rs]
    
    plt.figure()
    box = plt.boxplot([a, b], labels = ["Tip", "Shake"], whis = "range", patch_artist = True)
    for _, line_list in box.items():
        for i, line in enumerate(line_list):
            if i<(len(line_list)/2):
                line.set_color("blue")
                try:
                    line.set_facecolor("white")
                except:
                    pass
            else:
                line.set_color("red")
                try:
                    line.set_facecolor("white")
                except:
                    pass
    plt.ylabel("mahalanobis distribution", fontsize = 24)
    plt.xticks(fontsize = 24)
    plt.show()
    
    # plt.figure()
    # plt.boxplot([[np.std(s) for s in tkde_rs], [np.std(s) for s in skde_rs]], labels = ["Tip", "Shake"])
    # plt.show()


def ketchup_hist_comp():
    plt.rcParams["font.size"] = 24
    plt.rcParams["figure.figsize"] = (8,8)
    td =  np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/npdata/datotal.npy")
    sd = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal.npy")

    size1, size2, dtheta1, dtheta2 = 0.61,0.66,0.1,0.4
    
    x = np.linspace(0,0.55,100)
    fig = go.Figure()
    idx_s1 = idx_of_the_nearest(np.linspace(0.3,0.8,100), size1)
    idx_s2 = idx_of_the_nearest(np.linspace(0.3,0.8,100), size2)
    idx_d1 = idx_of_the_nearest(np.linspace(0.1,1,100), dtheta1)
    idx_d2 = idx_of_the_nearest(np.linspace(0.1,1,100), dtheta2)
    crop_area = td[100-idx_d2:100-idx_d1,idx_s1:idx_s2].flatten()
    fig.add_trace(go.Histogram(
        x = crop_area, xbins = dict(start=0,end=0.55,size=0.027),
        histnorm = "probability",
        name = "Tip"
    ))
   
    crop_area = sd[idx_s1:idx_s2]
    fig.add_trace(go.Histogram(
        x = crop_area, xbins = dict(start=0,end=0.55,size=0.028),
        histnorm = "probability",
        name = "Shake"
    ))
    
    fig.update_layout(
        bargap = 0.,
        barmode = "overlay",
    )
    fig.update_traces(
        opacity = 0.75
    )
    fig['layout']['xaxis']['title']['text'] = "y<sub>amount</sub>"
    fig['layout']['yaxis']['title'] = "Count (normalized)"
    fig['layout']['xaxis']['range'] = (-0.01,0.56)
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 40
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
        # width = 900,
        # width = 1500,
        width = 1600,
        height = 800,
    )
    fig.write_image(PICTURE_DIR+"ketchup/multimodal.svg")
    fig.show()
    

def nobounce_check():
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/"
    # logs = [
    #     "0_5",
    #     "5_10","10_15","15_16","16_17","17_18","18_19","19_20","20_29","29_30","30_40","40_50","50_60","60_70","70_80","80_90","90_100",
    # ]
    
    # datotal = []
    # for log in logs:
    #     path = basedir + log + "g0/pred_true_log.yaml"
    #     with open(path, "rb") as f:
    #         pred_true_log = yaml.load(f)
    #     for i in range(len(pred_true_log)):
    #         datotal.append(pred_true_log[i]["Ftip_amount"]["true_output"][0])
    #     print(path, len(pred_true_log))
    # # print(datotal)
    # # print(len(datotal))
    # datotal = np.array(datotal).reshape(100,100)[::-1]
    # check_or_create_dir("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_nobounce_smsz_dtheta2/npdata")
    # np.save("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_nobounce_smsz_dtheta2/npdata/datotal.npy", datotal)
    # datotal = []
    # path = basedir + "shake_nobounce_smsz/smsz0308g0/pred_true_log.yaml"
    # with open(path, "rb") as f:
    #     pred_true_log = yaml.load(f)
    # for i in range(len(pred_true_log)):
    #     datotal.append(pred_true_log[i]["Fshake_amount"]["true_output"][0])
    # check_or_create_dir("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/shake_nobounce_smsz/npdata")
    # datotal = np.array(datotal)
    # np.save("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/shake_nobounce_smsz/npdata/datotal.npy", datotal)
            
    # datotal = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_nobounce_smsz_dtheta2/npdata/datotal.npy")
    # np.save("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_nobounce_smsz_dtheta2/npdata/datotal.npy", datotal[::-1])
    # datotal = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_nobounce_smsz_dtheta2/npdata/datotal.npy")
    
    
    fig = go.Figure()
    fig.update_layout(
        width = 1200,
        height = 1000,
    )
    fig.add_trace(go.Heatmap(
        z = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_nobounce_smsz_dtheta2/npdata/datotal.npy"), 
        x = np.linspace(0.3,0.8,100), y = np.linspace(0.1,1,100)[::-1],
        colorscale = [
            [0, "rgb(0, 0, 255)"],
            [0.2727, "rgb(0, 255, 255)"],
            [0.5454, "rgb(0, 255, 0)"],
            [0.772, "rgb(255, 255, 0)"],
            [1, "rgb(255, 0, 0)"],
        ],
        zmin = 0, zmax = 0.55,
        colorbar=dict(
            title = "y<sub>amount</sub>",
            titleside="top", ticks="outside",
            # x = posx_set[col-1], y = posy_set[row-1],
            thickness=23, 
            # len = clength,
            tickfont = dict(color = "black"),
        ),
    ))
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['yaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['dtick'] = 0.2
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    # fig.show()
    check_or_create_dir(PICTURE_DIR+"nobounce")
    fig.write_image(PICTURE_DIR+"nobounce/tip.svg")
    
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/shake_nobounce_smsz/npdata/datotal.npy").reshape(1,100), 
        x = np.linspace(0.3,0.8,100), y = [1],
        colorscale = [
            [0, "rgb(0, 0, 255)"],
            [0.2727, "rgb(0, 255, 255)"],
            [0.5454, "rgb(0, 255, 0)"],
            [0.772, "rgb(255, 255, 0)"],
            [1, "rgb(255, 0, 0)"],
        ],
        zmin = 0, zmax = 0.55,
        colorbar=dict(
            title = "y<sub>amount</sub>",
            titleside="top", ticks="outside",
            # x = posx_set[col-1], y = posy_set[row-1],
            thickness=23, 
            # len = clength,
            tickfont = dict(color = "black"),
        ),
    ))
    fig['layout']['xaxis']['title'] = "size"
    fig['layout']['yaxis']['title'] = "dtheta"
    fig['layout']['yaxis']['dtick'] = 0.2
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    # fig.show()
    fig.write_image(PICTURE_DIR+"nobounce/shake.svg")
    
    
def detect_tip_edge(eval_tip):
    # eval_tip = np.maximum(eval_tip, -3)
    kernel_x = np.array([
        [0,-1,0],
        [0,0,0],
        [0,1,0]
    ])
    kernel_y = np.array([
        [0,0,0],
        [-1,0,1],
        [0,0,0]
    ])
    eval_tip_edge = np.sqrt(cv2.filter2D(eval_tip, -1, kernel_x)**2 + cv2.filter2D(eval_tip, -1, kernel_y)**2)

    return eval_tip_edge

    
def Run(ct, *args):
    # ketchup_check()
    # ketchup_curve()
    # ketchup_kde()
    # ketchup_kde_comp()
    # ketchup_hist_comp()
    # nobounce_check()
    # calc_rfunc()
    # test()
    # shake_rfunc_plot()
    # opttest_baseline()
    # pref = lambda ep: "ErLCB4/checkpoints/t{}/ch500".format(ep)
    # pref = lambda ep: "Er/t{}".format(ep)
    # pref = lambda ep: "GMM9Sig8LCB4/checkpoints/t{}/ch500".format(ep)
    # pref = lambda ep: "TMMSig8LCB4/checkpoints/t{}/ch500".format(ep)
    # pref = lambda ep: "GMM12Sig8LCB4/checkpoints/t{}/ch100".format(ep)
    # pref = lambda ep: "GMM12Sig8LCB4/checkpoints/t{}/u1add25".format(ep)
    # pref = lambda ep: "GMM12Sig8LCB4/checkpoints/t{}/u2add25".format(ep)
    # pref = lambda ep: "GMM12Sig8LCB4/checkpoints/t{}/u3add50".format(ep)
    # pref = lambda ep: "GMM12Sig8LCB4_def/checkpoints/t{}/u1add50".format(ep)
    # pref = lambda ep: "GMM12Sig8LCB4_def/checkpoints/t{}/u2add50".format(ep)
    # for ep in range(1,30):
    #     check(pref(ep), ver = 4)
        # curve(pref(ep), ver = 3)
        # datotal(pref(ep), ver = 3)
        # evaluation(pref(ep), ver = 3)
    # pref = lambda ep: "ErLCB4/checkpoints/t{}/ch500".format(ep)
    # pref = lambda ep: "GMM12Sig8LCB4/checkpoints/t{}/ch500".format(ep)
    # pref = lambda ep: "Er/t{}".format(ep)
    # for ep in range(1,100):
    #     check(pref(ep))
        # curve(pref(ep))
        # evaluation(pref(ep))
        # evaluation_checkpoint(pref(ep), 0.608, [500], no_ch = True)
        # datotal(pref(ep))
        # datotal_checkpoint(pref(ep), 0.608, [500], no_ch = True)
        # datotal_shake(pref(ep))
        # datotal_custom(pref(ep))
        # evaluation_custom(pref(ep))
        # comp_checkpoint(pref(ep), [500], no_ch = True)
    # check("TMMSig8/checkpoints/t1/ch500")
    # evaluation("withOptBug/GMMSig5LCB3/t17")
    # datotal("withOptBug/GMMSig5LCB3/t17")
    # datotal_custom("withOptBug/GMMSig5LCB3/t17")
    # datotal_shake("GMM9Sig8LCB4/checkpoints/t84/ch500")
    # datotal_shake_custom("GMM9Sig8LCB4/checkpoints/t84/ch500")
    # datotal_shake("GMM9Sig5LCB4/checkpoints/t27/ch500")
    # datotal_shake("GMM9Sig5LCB4/checkpoints/t83/ch500")
    # datotal_shake("TMMSig8LCB4/checkpoints/t67/ch500")
    # evaluation_custom("ErLCB4/checkpoints/t33/ch500")
    # comp_custom("ErLCB4/checkpoints/t33/ch500")
    # datotal_custom("ErLCB4/checkpoints/t33/ch500")
    # jpx("GMMSig5LCB3/t4")
    # opttest_comp("GMM9Sig5LCB3/checkpoints", 99, "ch500/")
    # opttest_comp("GMM9Sig5LCB6/checkpoints", 49, "ch500/")
    # opttest_comp("ErLCB4/checkpoints", 70, "ch500/")
    # opttest_comp("Er", 100, "")
    # opttest_comp("TMMSig8LCB4/checkpoints", 99, "ch500/")
    # opttest_comp_custom("TMMSig8LCB4/checkpoints", 99, "ch500/")
    # opttest_comp("TMMSig8/checkpoints", 45, "ch500/")
    # opttest_comp("GMM9Sig8LCB4/checkpoints", 99, "ch500/")
    # opttest_comp("GMM9Sig5LCB4/checkpoints", 99, "ch500/")
    # opttest_comp("GMM9Sig8LCB4/sort", 83, "ch500/")
    # opttest_comp("GMM9Sig8LCB4_chop/sort", 80, "ch500/")
    # opttest_comp("GMM11Sig8LCB4/checkpoints", 99, "ch500/")
    # opttest_comp("GMM12Sig8LCB4/checkpoints", 99, "ch500/")
    # opttest_comp_concat("", 99, "ch500/")
    # opttest_comp("GMM12Sig8LCB4/checkpoints", 30, ch='ch100/', ver=3)
    # opttest_comp("GMM12Sig10LCB4/checkpoints", 50, "ch500/")
    # opttest_comp("GMM12Sig12LCB4/checkpoints", 94, "ch500/")
    # opttest_comp("GMM12Sig8LCB4/checkpoints", 30, ch='ch100/', ver=4)
    # opttest_comp("GMM12Sig8LCB4/checkpoints", 30, ch='u3add50/', ver=4)
    # opttest_comp("GMM12Sig8LCB4_def/checkpoints", 30, ch='u2add50/', ver=4)
    # for i in range(1,31):
    #     curve("withOptBug/GMMSig5LCB3/t{}".format(i))
    # curve("GMM12Sig8LCB4/checkpoints/t4/ch500")
    # gmm_curve("GMM12Sig8LCB4/checkpoints/t4/ch500")
    # for i in [57]:
    #     curve("GMM9Sig8LCB4/sort/t{}/ch500".format(i))
    # predict()
    # gmm_test()
    # gmm_test2()
    # gmm_test3()
    # gmm_test4()
    # gmm_test5()
    # gmm_test5v2()
    # gmm_test6()
    # gmm_test7()
    # gmm_test8()
    # gmm_test9()
    # gmm_test10()
    # gmm_test11()
    # gmm_test12()
    # gmm_test12v2()
    # gmm_test12v3()
    # gmm_test13()
    # gmm_test14()
    # gmm_test14v2()
    # gmm_test15()
    # gmm_test15v2()
    # gmm_logcheck()
    # gmm_ep("GMM9Sig5LCB3/t1/update1000", 762)
    # comp_checkpoint("TMMSig8LCB3/checkpoints/t1", range(350,510,25))
    # evaluation_checkpoint("ErLCB4/checkpoints/t33", 0.6181, [500])
    # datotal_checkpoint("ErLCB4/checkpoints/t33", 0.6181, [500])
    # datotal_checkpoint_custom("ErLCB4/checkpoints/t55", 0.6181, [500])
    # name_func = lambda ep: "GMM9Sig8LCB4_without5963/checkpoints/t{}".format(ep)
    # for i in range(1,11):
    #     print(i)
    #     name = name_func(i)
    #     comp_checkpoint(name, range(25,510,25))
    #     evaluation_checkpoint(name, 0.608, range(25,510,25))
    #     datotal_checkpoint(name, 0.608, range(25,510,25))
    #     evaluation(name+"/ch500")
    # sample_hist("GMM9Sig8LCB4/checkpoints", 99, "ch500/")
    # sample_hist("TMMSig8LCB4/checkpoints", 99, "ch500/")
    # sample_hist("Er", 99, "")
    # sample_hist_comp()
    # evaluation_tip_comp()
    # evaluation_shake_comp()
    # gmm_tip_comp()
    # validate()
    # smsz_bar()
    # name_func = lambda i: "Er/t{}".format(i)
    # name_func = lambda i: "ErLCB4/checkpoints/t{}".format(i)
    # for i in [54]:
    #     print(i)
        # name = name_func(i)
        # check(name+"/ch500")
        # comp_custom(name+"/ch500")
        # comp_checkpoint(name, [500])
        # evaluation_checkpoint(name, 0.618, [500])
        # evaluation_checkpoint_custom(name, 0.618, [500])
        # datotal_checkpoint(name, 0.618, [500])
        # datotal_checkpoint_custom(name, 0.618, [500])
        # evaluation_checkpoint_custom_comp(name, 0.618, [500])
    #     evaluation(name+"/ch500")
    #     evaluation_custom(name+"/ch500")
    #     datotal_custom(name+"/ch500")
    # curve_custom()
    # hist_concat(None)
    # hist_concat2(None)
    opttest_comp_custom("ErLCB4/checkpoints", 99, "ch500/")