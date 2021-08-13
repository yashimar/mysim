from .setup import *


TIP = "tip"
SHAKE = "shake"

def setup_datotal(dm, logdir):
    print("Setup datotal")
    logpath = logdir+"datotal.pickle"
    datotal = dict()
    if os.path.exists(logpath):
        with open(logpath, mode="rb") as f:
            datotal = pickle.load(f)
    else:
        for skill in [TIP, SHAKE]:
            datotal[skill] = {
                TRUE: dm.datotal[skill][TRUE],
                NNMEAN: None,
                NNERR: None,
                NNSD: None,
            }
            model = dm.nnmodels[skill].model
            if skill == TIP:
                X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ]).astype(np.float32)
                nnmean = model.Forward(x_data = X, train = False).data.reshape(100,100)
                nnerr = model.ForwardErr(x_data = X, train = False).data.reshape(100,100)
                nnsd = np.array([np.sqrt(model.Predict(x = [dtheta2, smsz], with_var = True).Var[0,0].item()) for dtheta2 in dm.dtheta2 for smsz in dm.smsz]).reshape(100,100)
            else:
                X = np.array([[smsz] for smsz in dm.smsz ]).astype(np.float32)
                nnmean = model.Forward(x_data = X, train = False).data.reshape(100)
                nnerr = model.ForwardErr(x_data = X, train = False).data.reshape(100)
                nnsd = np.array([np.sqrt(model.Predict(x = [smsz], with_var = True).Var[0,0].item()) for smsz in dm.smsz]).reshape(100)
            
            datotal[skill][NNMEAN] = nnmean
            datotal[skill][NNERR] = nnerr
            datotal[skill][NNSD] = nnsd
        
        with open(logpath, mode="wb") as f:
            pickle.dump(datotal, f)
        
    return datotal


def setup_gmmpred(dm, logdir):
    print("Setup gmmpred")
    logpath = logdir+"gmmpred.pickle"
    gmmpred = dict()
    if os.path.exists(logpath):
        with open(logpath, mode="rb") as f:
            gmmpred = pickle.load(f)
    else:
        if len(dm.gmms.keys()) != 0:
            for skill in [TIP, SHAKE]:
                gmm = dm.gmms[skill]
                gmm.train()
                if skill == TIP:
                    X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ]).astype(np.float32)
                    gmmpred[skill] = dm.gmms[skill].predict(X).reshape((100,100))
                else:
                    X = np.array([[smsz] for smsz in dm.smsz ]).astype(np.float32)
                    gmmpred[skill] = dm.gmms[skill].predict(X).reshape((100))
        else:
            gmmpred[TIP] = np.zeros((100,100))
            gmmpred[SHAKE] = np.zeros(100)
        with open(logpath, mode="wb") as f:
            pickle.dump(gmmpred, f)
        
    return gmmpred


def setup_eval(dm, logdir):
    print("Setup evalution")
    logpath = logdir+"evaluation.pickle"
    evaluation = dict()
    if os.path.exists(logpath):
        with open(logpath, mode="rb") as f:
            evaluation = pickle.load(f)
    else:
        with open(logdir+"datotal.pickle", mode="rb") as f:
            datotal = pickle.load(f)
        with open(logdir+"gmmpred.pickle", mode="rb") as f:
            gmmpred = pickle.load(f)
        sd_gain = dm.sd_gain
        LCB_ratio = dm.LCB_ratio
        rmodel = Rmodel("Fdatotal_gentle")
        evaluation = dict()
        
        datotal_nnmean = datotal[TIP][NNMEAN]
        datotal_nnsd = datotal[TIP][NNSD]
        gmm = gmmpred[TIP]
        rnn_sm = np.array([[rmodel.Predict(x=[0.3, datotal_nnmean[idx_dtheta2, idx_smsz]], x_var=[0, (sd_gain*(datotal_nnsd[idx_dtheta2, idx_smsz] + gmm[idx_dtheta2, idx_smsz]))**2], with_var=True) for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        evaluation["tip_Er"] = np.array([[rnn_sm[idx_dtheta2, idx_smsz].Y.item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        evaluation["tip_Sr"] = np.sqrt([[rnn_sm[idx_dtheta2, idx_smsz].Var[0,0].item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
        evaluation[TIP] = evaluation["tip_Er"] - LCB_ratio*evaluation["tip_Sr"]
        
        datotal_nnmean = datotal[SHAKE][NNMEAN]
        datotal_nnsd = datotal[SHAKE][NNSD]
        gmm = gmmpred[SHAKE]
        rnn_sm = np.array([rmodel.Predict(x=[0.3, datotal_nnmean[idx_smsz]], x_var=[0, (sd_gain*(datotal_nnsd[idx_smsz] + gmm[idx_smsz]))**2], with_var=True) for idx_smsz in range(100)])
        evaluation[SHAKE] = np.array([rnn_sm[idx_smsz].Y.item() for idx_smsz in range(100)]) - LCB_ratio*np.sqrt([rnn_sm[idx_smsz].Var[0,0].item() for idx_smsz in range(100)])

        with open(logpath, mode="wb") as f:
            pickle.dump(evaluation, f)
    
    return evaluation


def setup_full(dm, logdir, recreate = False):
    if recreate:
        os.remove(logdir+"datotal.pickle")
        os.remove(logdir+"gmmpred.pickle")
        os.remove(logdir+"evaluation.pickle")
    datotal = setup_datotal(dm, logdir)
    gmmpred = setup_gmmpred(dm, logdir)
    evaluation = setup_eval(dm, logdir)
    
    return datotal, gmmpred, evaluation
    