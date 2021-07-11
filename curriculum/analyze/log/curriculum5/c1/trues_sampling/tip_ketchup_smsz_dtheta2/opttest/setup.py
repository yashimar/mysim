# coding: UTF-8
from core_tool import *
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .learn import *


K10MEAN = "k10_mean"
K10ERR = "k10_error"
NNMEAN = "NN_mean"
NNERR = "NN_error"
NNSD = "NN_sd"
JP1 = "|JumpPoint(G=1) - NNmean|"
JP2 = "|JumpPoint(G=2) - NNmean|"
JP1DIFF = "|JumpPoint - (NNmean+/-1NNerr)|"
JP2DIFF = "|JumpPoint - (NNmean+/-2NNerr)|"
Er = "Er"
Sr = "Sr"
Er_1LCB = "Er_1LCB"
Er_2LCB = "Er_2LCB"
ErJP1 = "ErJP1"
SrJP1 = "SrJP1"
ErJP1_1LCB = "ErJP1_1LCB"
ErJP1_2LCB = "ErJP1_2LCB"
ErJP1ADD = "ErJP1_add"
SrJP1ADD = "SrJP1_add"
ErJP1ADD_1LCB = "ErJP1_add_1LCB"
ErJP1ADD_2LCB = "ErJP1_add_2LCB"
ErJP2 = "ErJP2"
SrJP2 = "SrJP2"
ErJP2_1LCB = "ErJP2_1LCB"
ErJP2_2LCB = "ErJP2_2LCB"
ErJP2ADD = "ErJP2_add"
SrJP2ADD = "SrJP2_add"
ErJP2ADD_1LCB = "ErJP2_add_1LCB"
ErJP2ADD_2LCB = "ErJP2_add_2LCB"
LCB1 = "LCB1"
LCB2 = "LCB2"
BASE_DIR = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/"


def setup_datotal(dm, logdir):
    print("Setup datotal")
    logpath = logdir+"datotal.pickle"
    if os.path.exists(logpath):
        with open(logpath, mode="rb") as f:
            datotal = pickle.load(f)
    else:
        datotal = {
            TRUE: dm.datotal[TRUE],
            K10MEAN: np.load(BASE_DIR+"npdata/datotal_mean.npy"),
            K10ERR: np.abs(dm.datotal[TRUE] - np.load(BASE_DIR+"npdata/datotal_mean.npy")),
            NNMEAN: None,
            NNERR: None,
            NNSD: None,
        }
        nnmean, nnerr, nnsd = np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100))
        for idx_dtheta2, dtheta2 in enumerate(dm.dtheta2):
            for idx_smsz, smsz in enumerate(dm.smsz):
                print("datotal", idx_dtheta2, idx_smsz)
                x_in = [dtheta2, smsz]
                xdatota_for_Forward = dm.nnmodel.model.DataX[0:1]; xdatota_for_Forward[0, 0] = x_in[0]; xdatota_for_Forward[0, 1] = x_in[1] #Chainerのバグに対処するため
                nnmean[idx_dtheta2, idx_smsz] = dm.nnmodel.model.Forward(x_data = xdatota_for_Forward, train = False).data.item() #model.Predict(..., x_var=zero).Yと同じ
                nnerr[idx_dtheta2, idx_smsz] = dm.nnmodel.model.ForwardErr(x_data = xdatota_for_Forward, train = False).data.item()
                nnsd[idx_dtheta2, idx_smsz] = np.sqrt(dm.nnmodel.model.Predict(x = x_in, with_var = True).Var[0,0].item())
        datotal[NNMEAN] = nnmean
        datotal[NNERR] = nnerr
        datotal[NNSD] = nnsd
        
        with open(logpath, mode="wb") as f:
            pickle.dump(datotal, f)
        
    return datotal


def setup_gmmpred(dm, gmm_name_list, logdir):
    print("Setup gmmpred")
    with open(logdir+"datotal.pickle", mode="rb") as f:
        datotal = pickle.load(f)
        
    gmmpreds = dict()
    for gmm, name in gmm_name_list:
        logpath = logdir+"gmmpred_{}.pickle".format(name)
        if os.path.exists(logpath):
            with open(logpath, mode="rb") as f:
                gmmpred = pickle.load(f)
                gmmpreds[name] = gmmpred
        else:
            gmmpred = np.zeros((100,100))
            for idx_dtheta2, dtheta2 in enumerate(dm.dtheta2):
                for idx_smsz, smsz in enumerate(dm.smsz):
                    print(name, idx_dtheta2, idx_smsz)
                    gmmpred[idx_dtheta2, idx_smsz] = gmm.predict([dtheta2, smsz]).item()
            gmmpreds[name] = gmmpred
            with open(logpath, mode="wb") as f:
                pickle.dump(gmmpred, f)
        
    with open(logdir+"gmmpreds.pickle", mode="wb") as f:
        pickle.dump(gmmpreds, f)
        
    return gmmpreds


def setup_reward(dm, logdir):
    print("Setup reward")
    with open(logdir+"datotal.pickle", mode="rb") as f:
        datotal = pickle.load(f)
    with open(logdir+"gmmpreds.pickle", mode="rb") as f:
        gmmpreds = pickle.load(f)
        
    reward = defaultdict(lambda: np.zeros((100,100)))
    reward_nogmm = defaultdict(lambda: np.zeros((100,100)))
    reward_gmm_noadd = defaultdict(lambda: np.zeros((100,100)))
    reward_gmm_add = defaultdict(lambda: np.zeros((100,100)))
    calc_nogmm, calc_gmm_noadd, calc_gmm_add = True, defaultdict(lambda: True), defaultdict(lambda: True)
    
    if os.path.exists(logdir+"reward_nogmm.pickle"):
        calc_nogmm = False
        with open(logdir+"reward_nogmm.pickle", mode="rb") as f:
            reward_nogmm = pickle.load(f)
    if os.path.exists(logdir+"reward_gmm_noadd.pickle"):
        with open(logdir+"reward_gmm_noadd.pickle", mode="rb") as f:
            reward_gmm_noadd = pickle.load(f)
        for k in reward_gmm_noadd.keys():
            calc_gmm_noadd[k.split("~")[0]] = False
    if os.path.exists(logdir+"reward_gmm_add.pickle"):
        with open(logdir+"reward_gmm_add.pickle", mode="rb") as f:
            reward_gmm_add = pickle.load(f)
        for k in reward_gmm_add.keys():
            calc_gmm_add[k.split("~")[0]] = False
        
    for idx_dtheta2, dtheta2 in enumerate(dm.dtheta2):
        for idx_smsz, smsz in enumerate(dm.smsz):
            print("reward", idx_dtheta2, idx_smsz)
            if calc_nogmm:
                r = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[NNMEAN][idx_dtheta2, idx_smsz]], x_var=[0, datotal[NNSD][idx_dtheta2, idx_smsz]**2], with_var=True)
                rmean, rsd = r.Y.item(), np.sqrt(r.Var[0,0]).item()
                reward_nogmm[Er][idx_dtheta2, idx_smsz] = rmean
                reward_nogmm[Sr][idx_dtheta2, idx_smsz] = rsd
                reward_nogmm[Er+"_"+LCB1][idx_dtheta2, idx_smsz] = rmean - 1*rsd
                reward_nogmm[Er+"_"+LCB2][idx_dtheta2, idx_smsz] = rmean - 2*rsd
            
            for name, gmmpred in gmmpreds.items():
                if calc_gmm_noadd[name]:
                    rjp = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[NNMEAN][idx_dtheta2, idx_smsz]], x_var=[0, max(datotal[NNSD][idx_dtheta2, idx_smsz], gmmpred[idx_dtheta2, idx_smsz])**2], with_var=True)
                    rjpmean, rjpsd = rjp.Y.item(), np.sqrt(rjp.Var[0,0]).item()
                    reward_gmm_noadd["{}~{}_{}".format(name, Er, "noadd")][idx_dtheta2, idx_smsz] = rjpmean
                    reward_gmm_noadd["{}~{}_{}".format(name, Sr, "noadd")][idx_dtheta2, idx_smsz] = rjpsd
                    reward_gmm_noadd["{}~{}_{}_{}".format(name, Er, "noadd", LCB1)][idx_dtheta2, idx_smsz] = rjpmean - 1*rjpsd
                    reward_gmm_noadd["{}~{}_{}_{}".format(name, Er, "noadd", LCB2)][idx_dtheta2, idx_smsz] = rjpmean - 2*rjpsd

                if calc_gmm_add[name]:
                    rjp = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[NNMEAN][idx_dtheta2, idx_smsz]], x_var=[0, (datotal[NNSD][idx_dtheta2, idx_smsz]+gmmpred[idx_dtheta2, idx_smsz])**2], with_var=True)
                    rjpmean, rjpsd = rjp.Y.item(), np.sqrt(rjp.Var[0,0]).item()
                    reward_gmm_add["{}~{}_{}".format(name, Er, "add")][idx_dtheta2, idx_smsz] = rjpmean
                    reward_gmm_add["{}~{}_{}".format(name, Sr, "add")][idx_dtheta2, idx_smsz] = rjpsd
                    reward_gmm_add["{}~{}_{}_{}".format(name, Er, "add", LCB1)][idx_dtheta2, idx_smsz] = rjpmean - 1*rjpsd
                    reward_gmm_add["{}~{}_{}_{}".format(name, Er, "add", LCB2)][idx_dtheta2, idx_smsz] = rjpmean - 2*rjpsd

    reward.update(reward_nogmm)
    reward.update(reward_gmm_noadd)
    reward.update(reward_gmm_add)

    with open(logdir+"reward_nogmm.pickle", mode="wb") as f:
        pickle.dump(reward_nogmm, f)
    with open(logdir+"reward_gmm_noadd.pickle", mode="wb") as f:
        pickle.dump(reward_gmm_noadd, f)
    with open(logdir+"reward_gmm_add.pickle", mode="wb") as f:
        pickle.dump(reward_gmm_add, f)
    with open(logdir+"reward.pickle", mode="wb") as f:
        pickle.dump(reward, f)

    return reward


def Run(ct, *args):
    name_pref = "t0.1"
    name_list = [
        "t0.1/t1",
        "t0.1/t2",
        "t0.1/t3",
        "t0.1/t4",
        "t0.1/t5",
        "t0.1/t6",
        "t0.1/t7",
        "t0.1/t8",
        "t0.1/t9",
        "t0.1/t10",
        "t0.1/t11",
        "t0.1/t12",
        "t0.1/t13",
        "t0.1/t14",
        "t0.1/t15",
        "t0.1/t16",
        "t0.1/t17",
        "t0.1/t18",
        "t0.1/t19",
        "t0.1/t20",
    ]
    recreate = False

    for name in name_list:
        logdir = BASE_DIR + "opttest/logs/{}/".format(name)
        print(logdir)
        dm = Domain.load(logdir+"dm.pickle")
        Gerr1Sig002 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50], Gerr = 1.0)
        Gerr1Sig002.train()
        Gerr1Sig005 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
        Gerr1Sig005.train()
        # Gerr2Sig002 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50], Gerr = 2.0)
        # Gerr2Sig002.train()
        # Gerr2Sig005 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 2.0)
        # Gerr2Sig005.train()
        gmm_name_list = [
            (Gerr1Sig002, "Gerr1_Sig002"),
            (Gerr1Sig005, "Gerr1_Sig005"),
        ]
        
        setup_datotal(dm, logdir)
        setup_gmmpred(dm, gmm_name_list, logdir)
        setup_reward(dm, logdir)
        