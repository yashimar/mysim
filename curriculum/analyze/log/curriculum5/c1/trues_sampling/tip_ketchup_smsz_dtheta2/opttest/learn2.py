#coding: UTF-8
from learn import *
from .setup import *
from scipy.optimize import nnls


class Domain2(Domain):
    def optimize(self, smsz):
        est_nn_Er, est_nn_Sr = [], []
        for idx_dtheta2, dtheta2 in enumerate(self.dtheta2):
            x_in = [dtheta2, smsz]
            est_datotal = self.nnmodel.model.Predict(x = x_in, with_var=True)
            x_var = [0, est_datotal.Var[0].item()]
            est_nn = self.rmodel.Predict(x=[0.3, est_datotal.Y[0].item()], x_var= x_var, with_var=True)
            est_nn_Er.append(est_nn.Y.item())
            est_nn_Sr.append(np.sqrt(est_nn.Var[0,0]).item())
            
        if self.use_gmm:
            self.gmm.train(self.log["true_r_at_est_opt_dthtea2"])
            X = np.array([[dtheta2, smsz] for dtheta2 in self.dtheta2])
            eval_sd_gmm = self.gmm.predict(X)
            E = np.array(est_nn_Er) - self.LCB_ratio*(np.array(est_nn_Sr) + eval_sd_gmm)
        else:
            E = np.array(est_nn_Er) - self.LCB_ratio*np.array(est_nn_Sr)
        idx_est_opt_dtheta2 = np.argmax(E)
        est_opt_dtheta2 = self.dtheta2[idx_est_opt_dtheta2]
        est_datotal = self.nnmodel.model.Predict(x=[est_opt_dtheta2, smsz], with_var=True).Y[0].item()
        est_opt_Er = est_nn_Er[idx_est_opt_dtheta2]
        
        return idx_est_opt_dtheta2, est_opt_dtheta2, est_datotal, est_opt_Er, E
    
    # @classmethod
    # def summarize(self, log):
        


class GMM6(GMM5, object):
    def __init__(self, nnmodel, diag_sigma, lam = 0.0, Gerr = 1.0):
        super(GMM6, self).__init__(nnmodel, diag_sigma, lam, Gerr)
        
    def extract_jps(self, log_r, predefine = None):
        if predefine != None:
            self.jumppoints["X"] = predefine["X"]
            self.jumppoints["Y"] = predefine["Y"]
            return
            
        rmodel = Rmodel("Fdatotal_gentle")
        self.jumppoints = {"X": [], "Y": []}
        self.logr = []
        model = self.nnmodel.model
        for x, r in zip(model.DataX, log_r):
            est_datotal = model.Predict(x = x, with_var=True)
            est_datotal_mean = est_datotal.Y.item()
            est_datotal_var = [0, est_datotal.Var[0].item()]
            est_eval = rmodel.Predict(x = [0.3, est_datotal_mean], x_var = est_datotal_var, with_var = True)
            est_eval_mean = est_eval.Y.item()
            est_eval_sd = np.sqrt(est_eval.Var.item()).item()
            if r < (est_eval_mean - self.Gerr*est_eval_sd):
                jp = (est_eval_mean - self.Gerr*est_eval_sd) - r
                self.jumppoints["X"].append(x.tolist())
                self.jumppoints["Y"].append(jp)
                self.logr.append(r)
        if len(self.jumppoints["X"]) >= 2:
            _, unique_index = np.unique(self.jumppoints["X"], return_index = True, axis = 0)
            self.jumppoints["X"] = [self.jumppoints["X"][idx] for idx in unique_index]
            self.jumppoints["Y"] = [self.jumppoints["Y"][idx] for idx in unique_index]
            self.logr = [self.logr[idx] for idx in unique_index]
                
    def train(self, log_r, recreate = True): #引数でdiag_sigmaの初期値をリストで設定してはいけない(ミュータブル)
        if recreate:
            self.extract_jps(log_r)
        self.gc_concat = []
        self.w_concat = []
        Var = np.diag(self.diag_sigma)**2
        for jpx, jpy in zip(np.array(self.jumppoints["X"]), np.array(self.jumppoints["Y"])):
            self.gc_concat.append(
                lambda x,jpx=jpx,jpy=jpy: multivariate_normal.pdf(x,jpx,Var)*(1./multivariate_normal.pdf(jpx,jpx,Var))*jpy
            )
        y = np.array(self.jumppoints["Y"])
        X = np.array([[gc(x).item() for gc in self.gc_concat] for x in self.jumppoints["X"]])
        if len(X) == 0:
            self.w_concat = []
        else:
            # self.w_concat = np.linalg.inv(X.T.dot(X) + self.lam*np.eye(X.shape[0])).dot(X.T).dot(Y)
            self.w_concat, rnorm = nnls(X, y)
            

def test():
    name = "onpolicy/Er/t1"
    logdir = BASE_DIR + "opttest/logs/{}/".format(name)
    dm = Domain.load(logdir+"dm.pickle")
    log = dm.log["true_r_at_est_opt_dthtea2"]
    
    p = 3
    lam = 1e-3
    Var = np.diag([(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)])**2
    gmm = GMM6(dm.nnmodel, diag_sigma=[(1.0-0.1)/(100./p), (0.8-0.3)/(100./p)], lam = lam)
    gmm.train(log)
    # for x, y in zip(gmm.jumppoints["X"], gmm.jumppoints["Y"]):
    #     print(x, y)
    reward = setup_reward(dm, logdir, ["gmm",], [(1.0,1.0),])
    X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ])
    gmmpred = gmm.predict(X).reshape(100,100)
    er = reward["Er"]
    sr = reward["Sr"]
    jpx_idx = [[idx_of_the_nearest(dm.dtheta2, x[0]), idx_of_the_nearest(dm.smsz, x[1])] for x in np.array(gmm.jumppoints["X"])]
    jpx_er = np.array([er[idx[0],idx[1]] for idx in jpx_idx])
    jpx_sr = np.array([sr[idx[0],idx[1]] for idx in jpx_idx])
    jpx_gmm = np.array([gmmpred[idx[0],idx[1]] for idx in jpx_idx])
    linex = [[x,x] for x in np.array(gmm.jumppoints["X"])[:,1]]
    liney = [[y,y] for y in np.array(gmm.jumppoints["X"])[:,0]]
    linetr0 = [[a,b] for a, b in zip(gmm.logr, jpx_er - jpx_sr)]
    linetr = [[a,b] for a, b in zip(gmm.logr, jpx_er - jpx_sr - jpx_gmm)]
    linegmm =[[a,b] for a, b in zip(gmm.jumppoints["Y"], jpx_gmm)]

    diffcs = [
            [0, "rgb(0, 0, 0)"],
            [0.01, "rgb(255, 255, 0)"],
            [1, "rgb(255, 0, 0)"],
        ]
    fig = go.Figure()
    fig.update_layout(
            hoverdistance = 2,
        )
    fig.add_trace(go.Surface(
        z = er, x = dm.smsz, y = dm.dtheta2,
            cmin = -8, cmax = 0, colorscale = "Viridis",
            colorbar = dict(
            len = 0.2,
        ),
        showlegend = False,
    ))
    for tz,tx,ty in zip(linetr0, linex, liney):
        fig.add_trace(go.Scatter3d(
            z = tz, x = tx, y = ty,
            mode = "lines",
            line = dict(
                color = "red",
            ),
            showlegend = False,
        ))
    fig['layout']['scene']['zaxis_autorange'] = 'reversed'
    
    # fig.add_trace(go.Surface(
    #     z = er - sr - gmmpred, x = dm.smsz, y = dm.dtheta2,
    #         cmin = -8, cmax = 0, colorscale = "Viridis",
    #         colorbar = dict(
    #         len = 0.2,
    #     ),
    #     showlegend = False,
    # ))
    fig.add_trace(go.Scatter3d(
        z = gmm.logr, x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
        mode = "markers",
        showlegend = False,
            marker = dict(
            color = "red",
            size = 4,
        )
    ))
    # fig['layout']['scene']['zaxis_autorange'] = 'reversed'
    # for tz,tx,ty in zip(linetr, linex, liney):
    #     fig.add_trace(go.Scatter3d(
    #         z = tz, x = tx, y = ty,
    #         mode = "lines",
    #         line = dict(
    #             color = "red",
    #         ),
    #         showlegend = False,
    #     ))
    
    # fig.add_trace(go.Surface(
    #     z = gmmpred, x = dm.smsz, y = dm.dtheta2,
    #     colorscale = diffcs,
    #         colorbar = dict(
    #         len = 0.2,
    #     ),
    # ))
    # fig.add_trace(go.Scatter3d(
    #     z = gmm.jumppoints["Y"], x = np.array(gmm.jumppoints["X"])[:,1], y = np.array(gmm.jumppoints["X"])[:,0],
    #     mode = "markers",
    #     showlegend = False,
    #         marker = dict(
    #         color = "red",
    #         size = 4,
    #     )
    # ))
    # for tz,tx,ty in zip(linegmm, linex, liney):
    #     fig.add_trace(go.Scatter3d(
    #         z = tz, x = tx, y = ty,
    #         mode = "lines",
    #         line = dict(
    #             color = "red",
    #         ),
    #         showlegend = False,
    #     ))
    fig.show()
    

def Run(ct, *args):
    test()