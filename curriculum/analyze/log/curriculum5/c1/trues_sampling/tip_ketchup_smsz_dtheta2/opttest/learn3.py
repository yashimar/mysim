#coding: UTF-8
from .learn2 import *
from .setup2 import *
import cma
from scipy.optimize import fmin_l_bfgs_b
from scipy import optimize
from scipy.special import gammaln
import shutil
from math import pi, sqrt, gamma


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
    def __init__(self, nnmodel, diag_sigma, options, lam = 0.0, Gerr = 1.0):
        super(GMM8, self).__init__(nnmodel, diag_sigma, Gerr)
        self.options = options
    
    def opt_funciton(self, w, X, y, tau, lam):
        w = np.maximum(0,w) #wの定義域をせいにするため
        delta = y - X.dot(w)
        indic = np.array(delta <= 0., dtype=np.float32)
        rho = (tau - indic)*delta
        
        return sum(rho) + lam * w.T.dot(w)
    
    def train(self, recreate_jp = True): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        tau = self.options['tau']
        lam = self.options['lam']
        maxiter = self.options['maxiter']
        verbose = self.options['verbose']
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
        elif len(X) == 1:
            self.w_concat = [1]
        else:
            mean_init = [1.0] * len(y)
            sd_init = 0.5
            es = cma.CMAEvolutionStrategy(mean_init, sd_init, {'seed':234, 'maxiter': maxiter})
            f = lambda w: self.opt_funciton(w, X, y, tau, lam)
            es.optimize(f, verbose = verbose)
            self.w_concat = np.maximum(0, es.result[0]) #opt_funcでwにrelu変換を施しているため


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
                lambda x,jpx=jpx,jpy=jpy: multivariate_normal.pdf(x,jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy
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
            
    def predict(self, x):
        if len(np.array(x).shape) == 1:
            x = [x]
        pred = np.zeros((len(x)))
        for gc, w in zip(self.gc_concat, self.w_concat):
            pred += w*gc(x)
        pred = np.maximum(0, pred)
        return pred
    
    
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
        
            
def gmm_test():
    pref = "onpolicy2/Er/"
    logdir = BASE_DIR + "opttest/logs/{}/t9/".format(pref)
    dm = Domain2.load(logdir+"dm.pickle")
    
    p = 5
    lam = 1e-5
    Var = np.diag([(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)])**2
    # gmm = GMM5(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], lam = 1e-4)
    # gmm = GMM4(dm.nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)])
    # gmm = GMM7(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)])
    gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./5), (0.8-0.3)/(100./5)], options = {"tau": 0.9, "lam": 1e-6})
    # gmm = TMM(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./5), (0.8-0.3)/(100./5)], options = {"tau": 0.9, "lam": 1e-6})
    gmm.extract_jps()
    gmm.jumppoints.update({"X": [[0., x[1]] for x in gmm.jumppoints["X"]]})
    tx = np.array([x[1] for x in gmm.jumppoints["X"]])
    uq = np.unique(tx, return_index=True)[1][[0,1,2,7,8,9,10,11,12]]
    # uq = np.unique(tx, return_index=True)[1]
    tx = np.array([x[1] for x in gmm.jumppoints["X"]])
    gmm.jumppoints.update({"X": [x for i, x in enumerate(gmm.jumppoints["X"]) if i in uq]})
    gmm.jumppoints.update({"Y": [y for i, y in enumerate(gmm.jumppoints["Y"]) if i in uq]})
    gmm.train(recreate_jp = False)
    print(gmm.w_concat)
    
    x_list = np.linspace(0.3, 0.8, 1000)
    X = np.array([[0., x] for x in x_list])
    P = gmm.predict(X)
    
    ttx = np.array(gmm.jumppoints["X"])
    tty = np.array(gmm.jumppoints["Y"])
    
    jpmeta = [[(multivariate_normal.pdf([0.,x],jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy).item() for x in x_list] for jpx, jpy in zip(ttx, tty)]
    # jpx = ttx[0]
    # jpy = tty[0]
    # for x in x_list:
    #     tmp = multivariate_normal.pdf([0., x],jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy
    #     tmp = tmp.item()
    #     print(x, tmp)
    # # print(jpmeta[0])
    # print(jpx)
    # jpge
    
    tttx = [x[1] for x in ttx]
    fig = plt.figure()
    for jp in jpmeta:
        plt.scatter(x_list, jp, c="skyblue")
    plt.scatter(x_list, P, c="purple")
    plt.scatter(tttx, tty, c="orange")
    # plt.scatter(tttx, gmm.predict(ttx))
    plt.xlim(0.3,0.65)
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
            "est_gmm_JP_tip": [],
            "est_gmm_JP_shake": [],
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
        

def shake_rfunc_plot():
    datotal = np.load("/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/curriculum5/c1/trues_sampling/shake_ketchup_smsz/npdata/datotal.npy")
    r = []
    for d in datotal:
        # r.append(rfunc(d))
        r.append(d)
    fig = plt.figure()
    plt.scatter(x = np.linspace(0.3,0.8,100), y = r)
    # plt.hlines(xmin=0.3,xmax=0.8,y=-1,color="red",linestyle="dashed")
    plt.hlines(xmin=0.3,xmax=0.8,y=0.3,color="red",linestyle="dashed")
    plt.show()
    
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
            
            
def execute_checkpoint(base_logdir, sd_gain, LCB_ratio, gmm_lams, num_ep, num_rand_sample, num_learn_step, num_checkpoints):
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
            dm = Domain3(new_logdir, sd_gain, LCB_ratio)
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


def opttest_comp(name, n, ch = None):
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
    for p in [0,2,5,10,50,90,95,98,100]:
        yp[p] = np.percentile(y_concat, p, axis = 0)
        for skill in[TIP, SHAKE]:
            yestp[skill][p] = np.percentile(np.array(yest_concat[skill]), p, axis = 0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yp[50],
        mode = "markers",
        name = "reward at opt param (0%, 100%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yp[100]-yp[50],
            arrayminus=yp[50]-yp[0],
            thickness=0.8,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yp[50],
        mode = "markers",
        name = "reward at opt param (2%, 98%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yp[98]-yp[50],
            arrayminus=yp[50]-yp[2],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
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
    fig['layout']['yaxis']['title'] = "reward / evaluation"
    plotly.offline.plot(fig, filename = save_img_dir + "opttest_comp.html", auto_open=False)
    
    
def comp_checkpoint(name, ep_checkpoints):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    
    trace = defaultdict(list)
    for ep in ep_checkpoints:
        print(ep)
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
    fig['layout']['yaxis']['title'] = "reward / evaluation"
    fig['layout']['yaxis']['range'] = (-8,0.5)
    for ep_idx, ep in enumerate(ep_checkpoints):
        fig['layout']['sliders'][0]['steps'][ep_idx]['label'] = ep
    plotly.offline.plot(fig, filename = save_img_dir + "comp.html", auto_open=False)


def check(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    # name = "GMMSig5LCB3/t1"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
    true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
    true_yshake = dm.datotal[SHAKE][RFUNC]
    est_ytip = np.max(evaluation[TIP], axis=0)
    est_yshake = evaluation[SHAKE]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = log["ep"], y = log["r_at_est_optparam"],
        mode = "markers",
        text = ["<br />".join(["ep: {}".format(ep), "smsz: {}".format(smsz), "optparam: {}".format(optparam), "opteval: {}".format(opteval), "skill: {}".format(skill)]) for ep, smsz, optparam, opteval, skill in zip(log["ep"], log["smsz"], log["est_optparam"], log["opteval"], log["skill"])],
    ))
    fig['layout']['xaxis']['title'] = "episode"
    fig['layout']['yaxis']['title'] = "reward"
    plotly.offline.plot(fig, filename = save_img_dir + "hist.html", auto_open=False)
    
    for ep in range(50,len(log["ep"])):
        if log["r_at_est_optparam"][ep] < -1:
            print(ep, log["smsz"][ep], log["r_at_est_optparam"][ep])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = true_ytip,
        mode = "markers",
        name = "reward (tip) at est optparam"
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = true_yshake,
        mode = "markers",
        name = "reward (shake) at est optparam"
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = est_ytip,
        mode = "markers",
        name = "evaluatioin (tip) at est optparam"
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = est_yshake,
        mode = "markers",
        name = "evaluation (shake) at est optparam"
    ))
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "reward / evaluation"
    fig['layout']['yaxis']['range'] = (-5,0.2)
    plotly.offline.plot(fig, filename = save_img_dir + "comp.html", auto_open=False)
    
    rc = dm.datotal[TIP][RFUNC].reshape(100*100)
    idx = [i for i,r in enumerate(rc) if r<-0.7]
    rc = rc[idx]
    smsz = np.array([smsz for smsz in dm.smsz]*100)[idx]
    dtheta2 = np.array(sum([[dtheta2]*100 for dtheta2 in dm.dtheta2],[]))[idx]
    
    n_row = 3
    clength = 0.2
    fig = make_subplots(
        rows=n_row, cols=2, 
        subplot_titles=["datotal 生データ (100×100)", "報酬 生データ (100×100)", 
                        "飛び値モデル (真値報酬-0.7以下の地点プロット)", "評価関数 (真値報酬-0.7以下の地点プロット)", 
                        "飛び値モデル", "評価関数", 
                        ],
        horizontal_spacing = 0.1,
        vertical_spacing = 0.05,
    )
    fig.update_layout(
        height=600*n_row, width=1750, 
        margin=dict(t=100,b=150),
        hoverdistance = 2,
    )
    diffcs = [
        [0, "rgb(255, 255, 255)"],
        [0.01, "rgb(255, 255, 200)"],
        [1, "rgb(255, 0, 0)"],
    ]
    z_rc_pos_scale_cs_scatterz_scatterscale_set = (
        (datotal[TIP][TRUE], 1, 1, 0.46, 0.94, 0., 0.55, None, None, None, None), (dm.datotal[TIP][RFUNC], 1, 2, 0.46, 0.94, -3, 0., None, None, None, None),
        # (gmmpred[TIP], 2, 1, 0.46, 0.28, 0., 0.2, diffcs, "badr", -3, 0), (evaluation[TIP], 2, 2, 0.46, 0.94, -3, 0., None, "badr", -3, 0),
        (gmmpred[TIP], 2, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), (evaluation[TIP], 2, 2, 0.46, 0.94, -3, 0., None, "badr", None, None),
        (gmmpred[TIP], 3, 1, 0.46, 0.28, 0., 0.2, diffcs, None, None, None), (evaluation[TIP], 3, 2, 0.46, 0.94, -3, 0., None, None, None, None),
    )
    posx_set = [0.46, 1.0075]
    posy_set = (lambda x: [0.1 + 0.7/(x-1)*i for i in range(x)][::-1])(n_row)
    for z, row, col, posx, posy, zmin, zmax, cs, scz, sczmin, sczmax in z_rc_pos_scale_cs_scatterz_scatterscale_set:
        if np.sum(z) != 0:
            fig.add_trace(go.Heatmap(
                z = z, x = dm.smsz, y = dm.dtheta2,
                colorscale = cs if cs != None else "Viridis",
                zmin = zmin, zmax = zmax,
                colorbar=dict(
                    titleside="top", ticks="outside",
                    x = posx_set[col-1], y = posy_set[row-1],
                    thickness=23, len = clength,
                ),
            ), row, col)
            if scz != "badr": continue
            fig.add_trace(go.Scatter(
                x = smsz, y = dtheta2,
                mode='markers',
                showlegend = False,
                marker = dict(
                    size = 4,
                    color = rc,
                    colorscale = "Viridis",
                    cmin = sczmin,
                    cmax = sczmax,
                    line = dict(
                        color = "black",
                        width = 1,
                    )
                ),
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
                    ),
                ),
            ), row, col)
    for i in range(1,len(z_rc_pos_scale_cs_scatterz_scatterscale_set)+1):
        fig['layout']['xaxis'+str(i)]['title'] = "size_srcmouth"
        fig['layout']['yaxis'+str(i)]['title'] = "dtheta2"
    plotly.offline.plot(fig, filename = save_img_dir + "heatmap.html", auto_open=False)
    

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
    
    gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-6})
    # gmm = TMM(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./8), (0.8-0.3)/(100./8)], options = {"tau": 0.9, "lam": 1e-6})
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
                                color= "blue" if addv == 0 else "grey", 
                                size=8,
                                symbol="x",
                            ),
                    visible=False,
                ))
            else:
                trace[2+i].append(go.Scatter(x=[], y=[]))
        for i,addv in enumerate(range(-4,5)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    trace[3+i].append(go.Scatter(
                            x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list]],
                            mode='markers', 
                            name="Obs {:.3f}".format(tmp_smsz),
                            marker=dict(
                                color= "purple" if addv == 0 else "pink", 
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
    fig['layout']['yaxis']['title'] = "return"
    fig['layout']['yaxis']['range'] = (-8,0.5)
    for smsz_idx, smsz in enumerate(dm.smsz):
        fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "evaluation_tip_fixgmm.html", auto_open=False)

   
def evaluation(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
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
                                color= "blue" if addv == 0 else "grey", 
                                size=8,
                                symbol="x",
                            ),
                    visible=False,
                ))
            else:
                trace[2+i].append(go.Scatter(x=[], y=[]))
        for i,addv in enumerate(range(-4,5)):
            if 0<=(smsz_idx+addv)<=(len(dm.smsz)-1):
                tmp_smsz = dm.smsz[smsz_idx+addv]
                if tmp_smsz in dm.log["smsz"]:
                    log_smsz_idx_list = [log_smsz_idx for log_smsz_idx, (log_smsz, skill) in enumerate(zip(dm.log["smsz"], dm.log["skill"])) if (log_smsz == tmp_smsz) and (skill == TIP)]
                    trace[3+i].append(go.Scatter(
                            x=np.array(dm.log["est_optparam"])[log_smsz_idx_list], y=[rfunc(_datotal) for _datotal in np.array(dm.log["datotal_at_est_optparam"])[log_smsz_idx_list]],
                            mode='markers', 
                            name="Obs {:.3f}".format(tmp_smsz),
                            marker=dict(
                                color= "purple" if addv == 0 else "pink", 
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
    fig['layout']['yaxis']['title'] = "return"
    fig['layout']['yaxis']['range'] = (-8,0.5)
    for smsz_idx, smsz in enumerate(dm.smsz):
        fig['layout']['sliders'][0]['steps'][smsz_idx]['label'] = round(smsz,4)
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "evaluation_tip.html", auto_open=False)


def evaluation_checkpoint(name, smsz, ep_checkpoints):
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
        datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        est_ytip = np.max(evaluation[TIP], axis=0)
        est_yshake = evaluation[SHAKE]
        
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
        for i,addv in enumerate(range(-4,5)):
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
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta2"
    fig['layout']['yaxis']['title'] = "return"
    fig['layout']['yaxis']['range'] = (-5,0.5)
    for ep_idx, ep in enumerate(ep_checkpoints):
        fig['layout']['sliders'][0]['steps'][ep_idx]['label'] = ep
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "{}.html".format(str(smsz).split('.')[1]), auto_open=False)


def datotal(name):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy2/{}/".format(name)
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
    
    gmm = GMM9(dm.nnmodels[TIP], diag_sigma=[(1.0-0.1)/(100./5), (0.8-0.3)/(100./5)], options = {"tau": 0.9, "lam": 1e-6})
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
    plotly.offline.plot(fig, filename = save_img_dir + "datotal_tip_fixgmm.html", auto_open=False)
 

def datotal_checkpoint(name, smsz, ep_checkpoints):
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
            marker=dict(color="pink"),
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
    fig.update_layout(
        sliders=sliders
    )
    fig['layout']['xaxis']['title'] = "dtheta2"
    fig['layout']['yaxis']['title'] = "datotal"
    fig['layout']['yaxis']['range'] = (-0.05,0.6)
    for ep_idx, ep in enumerate(ep_checkpoints):
        fig['layout']['sliders'][0]['steps'][ep_idx]['label'] = ep
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "{}.html".format(str(smsz).split('.')[1]), auto_open=False)


def curve(name):
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
    plotly.offline.plot(fig, filename = save_img_dir + "curve_tip.html", auto_open=False)
    

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

    
def Run(ct, *args):
    # test()
    # shake_rfunc_plot()
    # pref = lambda ep: "GMM9Sig5LCB3/checkpoints/t{}/ch500".format(ep)
    # pref = lambda ep: "Er/t{}".format(ep)
    # for ep in range(1,2):
    #     check(pref(ep))
    # check("TMMSig8/checkpoints/t1/ch500")
    # evaluation("withOptBug/GMMSig5LCB3/t17")
    # datotal("withOptBug/GMMSig5LCB3/t17")
    # datotal_custom("withOptBug/GMMSig5LCB3/t17")
    evaluation_custom("GMM9Sig5LCB3/checkpoints/t1/ch500")
    # jpx("GMMSig5LCB3/t4")
    # opttest_comp("GMM9Sig5LCB3/checkpoints", 99, "ch500/")
    # opttest_comp("GMM9Sig5LCB6/checkpoints", 49, "ch500/")
    # opttest_comp("ErLCB4/checkpoints", 70, "ch500/")
    # opttest_comp("Er", 100, "")
    # opttest_comp("TMMSig8LCB4/checkpoints", 99, "ch500/")
    # opttest_comp("TMMSig8/checkpoints", 45, "ch500/")
    # for i in range(1,31):
    #     curve("withOptBug/GMMSig5LCB3/t{}".format(i))
    # curve("GMM9Sig5LCB3/checkpoints/t1/ch500")
    # predict()
    # gmm_test()
    # gmm_test2()
    # gmm_ep("GMM9Sig5LCB3/t1/update1000", 762)
    # comp_checkpoint("TMMSig8LCB3/checkpoints/t1", range(350,510,25))
    # evaluation_checkpoint("ErLCB4/checkpoints/t3", 0.6181, [500])
    # datotal_checkpoint("ErLCB4/checkpoints/t3", 0.6181, [500])
    