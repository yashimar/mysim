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
                print("datotal", logdir, idx_dtheta2, idx_smsz)
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


def setup_obsr(dm, obsr_name_list, logdir):
    print("Setup obsr")
    obsrcalcs = dict()
    for obsr, name in obsr_name_list:
        logpath = logdir+"obsrcalc_{}.pickle".format(name)
        if os.path.exists(logpath):
            with open(logpath, mode="rb") as f:
                obsrcalc = pickle.load(f)
            obsrcalcs[name] = obsrcalc
        else:
            X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ])
            obsrcalc = obsr.calc_reward(X)
            obsrcalc = obsrcalc.reshape(100,100)
            obsrcalcs[name] = obsrcalc
            with open(logpath, mode="wb") as f:
                pickle.dump(obsrcalc, f)
        
    with open(logdir+"obsrcalcs.pickle", mode="wb") as f:
        pickle.dump(obsrcalcs, f)
        
    return obsrcalcs


def setup_unobssd(dm, unobs_name_list, logdir):
    print("Setup unobssd")
    unobssds = dict()
    for us, name in unobs_name_list:
        logpath = logdir+"obsrcalc_{}.pickle".format(name)
        if os.path.exists(logpath):
            with open(logpath, mode="rb") as f:
                unobssd = pickle.load(f)
            unobssds[name] = unobssd
        else:
            X = np.array([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ])
            unobssd = us.calc_sd(X)
            unobssd = unobssd.reshape(100,100)
            unobssds[name] = unobssd
            with open(logpath, mode="wb") as f:
                pickle.dump(unobssd, f)
        
    with open(logdir+"unobssds.pickle", mode="wb") as f:
        pickle.dump(unobssds, f)
        
    return unobssds


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
                    print(name, logdir, idx_dtheta2, idx_smsz)
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
    with open(logdir+"unobssds.pickle", mode="rb") as f:
        unobssds = pickle.load(f)
    with open(logdir+"gmmpreds.pickle", mode="rb") as f:
        gmmpreds = pickle.load(f)
        
    reward = defaultdict(lambda: np.zeros((100,100)))
    reward_normal = defaultdict(lambda: np.zeros((100,100)))
    reward_gmm_noadd = defaultdict(lambda: np.zeros((100,100)))
    reward_gmm_add = defaultdict(lambda: np.zeros((100,100)))
    reward_unobs = defaultdict(lambda: np.zeros((100,100)))
    reward_unobs_gmm_add = defaultdict(lambda: np.zeros((100,100)))
    calc_normal = True
    calc_gmm_noadd, calc_gmm_add = defaultdict(lambda: True), defaultdict(lambda: True)
    calc_unobs = defaultdict(lambda: True)
    calc_unobs_gmm_add = defaultdict(lambda: True)
    
    if os.path.exists(logdir+"reward_normal.pickle"):
        calc_normal = False
        with open(logdir+"reward_normal.pickle", mode="rb") as f:
            reward_normal = pickle.load(f)
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
    if os.path.exists(logdir+"reward_unobs.pickle"):
        with open(logdir+"reward_unobs.pickle", mode="rb") as f:
            reward_unobs = pickle.load(f)
        for k in reward_unobs.keys():
            calc_unobs[k.split("~")[0]] = False
    if os.path.exists(logdir+"reward_unobs_gmm_add.pickle"):
        with open(logdir+"reward_unobs_gmm_add.pickle", mode="rb") as f:
            reward_unobs_gmm_add = pickle.load(f)
        for k in reward_unobs_gmm_add.keys():
            calc_unobs_gmm_add[k.split("~")[0]] = False
        
    for idx_dtheta2, dtheta2 in enumerate(dm.dtheta2):
        for idx_smsz, smsz in enumerate(dm.smsz):
            print("reward", logdir, idx_dtheta2, idx_smsz)
            if calc_normal:
                r = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[NNMEAN][idx_dtheta2, idx_smsz]], x_var=[0, datotal[NNSD][idx_dtheta2, idx_smsz]**2], with_var=True)
                rmean, rsd = r.Y.item(), np.sqrt(r.Var[0,0]).item()
                reward_normal[Er][idx_dtheta2, idx_smsz] = rmean
                reward_normal[Sr][idx_dtheta2, idx_smsz] = rsd
                reward_normal[Er+"_"+LCB1][idx_dtheta2, idx_smsz] = rmean - 1*rsd
                reward_normal[Er+"_"+LCB2][idx_dtheta2, idx_smsz] = rmean - 2*rsd
            
            for name, gmmpred in gmmpreds.items():
                # if calc_gmm_noadd[name]:
                #     rjp = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[NNMEAN][idx_dtheta2, idx_smsz]], x_var=[0, max(datotal[NNSD][idx_dtheta2, idx_smsz], gmmpred[idx_dtheta2, idx_smsz])**2], with_var=True)
                #     rjpmean, rjpsd = rjp.Y.item(), np.sqrt(rjp.Var[0,0]).item()
                #     reward_gmm_noadd["{}~{}_{}".format(name, Er, "noadd")][idx_dtheta2, idx_smsz] = rjpmean
                #     reward_gmm_noadd["{}~{}_{}".format(name, Sr, "noadd")][idx_dtheta2, idx_smsz] = rjpsd
                #     reward_gmm_noadd["{}~{}_{}_{}".format(name, Er, "noadd", LCB1)][idx_dtheta2, idx_smsz] = rjpmean - 1*rjpsd
                #     reward_gmm_noadd["{}~{}_{}_{}".format(name, Er, "noadd", LCB2)][idx_dtheta2, idx_smsz] = rjpmean - 2*rjpsd

                if calc_gmm_add[name]:
                    rjp = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[NNMEAN][idx_dtheta2, idx_smsz]], x_var=[0, (datotal[NNSD][idx_dtheta2, idx_smsz]+gmmpred[idx_dtheta2, idx_smsz])**2], with_var=True)
                    rjpmean, rjpsd = rjp.Y.item(), np.sqrt(rjp.Var[0,0]).item()
                    reward_gmm_add["{}~{}_{}".format(name, Er, "add")][idx_dtheta2, idx_smsz] = rjpmean
                    reward_gmm_add["{}~{}_{}".format(name, Sr, "add")][idx_dtheta2, idx_smsz] = rjpsd
                    reward_gmm_add["{}~{}_{}_{}".format(name, Er, "add", LCB1)][idx_dtheta2, idx_smsz] = rjpmean - 1*rjpsd
                    reward_gmm_add["{}~{}_{}_{}".format(name, Er, "add", LCB2)][idx_dtheta2, idx_smsz] = rjpmean - 2*rjpsd
                    
            for name, unobssd in unobssds.items():
                if calc_unobs[name]:
                    r = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[NNMEAN][idx_dtheta2, idx_smsz]], x_var=[0, (datotal[NNSD][idx_dtheta2, idx_smsz]+unobssd[idx_dtheta2, idx_smsz])**2], with_var=True)
                    rmean, rsd = r.Y.item(), np.sqrt(r.Var[0,0]).item()
                    reward_unobs["{}~{}".format(name, Er)][idx_dtheta2, idx_smsz] = rmean
                    reward_unobs["{}~{}".format(name, Sr)][idx_dtheta2, idx_smsz] = rsd
                    reward_unobs["{}~{}_{}".format(name, Er, LCB1)][idx_dtheta2, idx_smsz] = rmean - 1*rsd
                    reward_unobs["{}~{}_{}".format(name, Er, LCB2)][idx_dtheta2, idx_smsz] = rmean - 2*rsd
                    
            for n_unobs, unobssd in unobssds.items():
                for n_gmm, gmmpred in gmmpreds.items():
                    if calc_unobs_gmm_add["{}-{}".format(n_unobs, n_gmm)]:
                        r = Rmodel("Fdatotal_gentle").Predict(x=[0.3, datotal[NNMEAN][idx_dtheta2, idx_smsz]], x_var=[0, (datotal[NNSD][idx_dtheta2, idx_smsz]+unobssd[idx_dtheta2, idx_smsz]+gmmpred[idx_dtheta2, idx_smsz])**2], with_var=True)
                        rmean, rsd = r.Y.item(), np.sqrt(r.Var[0,0]).item()
                        reward_unobs_gmm_add["{}-{}~{}".format(n_unobs, n_gmm, Er)][idx_dtheta2, idx_smsz] = rmean
                        reward_unobs_gmm_add["{}-{}~{}".format(n_unobs, n_gmm, Sr)][idx_dtheta2, idx_smsz] = rsd
                        reward_unobs_gmm_add["{}-{}~{}_{}".format(n_unobs, n_gmm, Er, LCB1)][idx_dtheta2, idx_smsz] = rmean - 1*rsd
                        reward_unobs_gmm_add["{}-{}~{}_{}".format(n_unobs, n_gmm, Er, LCB2)][idx_dtheta2, idx_smsz] = rmean - 2*rsd

    reward.update(reward_normal)
    reward.update(reward_gmm_noadd)
    reward.update(reward_gmm_add)
    reward.update(reward_unobs)
    reward.update(reward_unobs_gmm_add)

    with open(logdir+"reward_normal.pickle", mode="wb") as f:
        pickle.dump(reward_normal, f)
    with open(logdir+"reward_gmm_noadd.pickle", mode="wb") as f:
        pickle.dump(reward_gmm_noadd, f)
    with open(logdir+"reward_gmm_add.pickle", mode="wb") as f:
        pickle.dump(reward_gmm_add, f)
    with open(logdir+"reward_unobs.pickle", mode="wb") as f:
        pickle.dump(reward_unobs, f)
    with open(logdir+"reward_unobs_gmm_add.pickle", mode="wb") as f:
        pickle.dump(reward_unobs_gmm_add, f)
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
        dm = Domain.load(logdir+"dm.pickle")
        observations = np.array([
            dm.log["est_opt_dtheta2"],
            dm.log["smsz"]
        ]).T
        # G1ObsrSig001 = ObservationReward(observations, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100])
        # G1ObsrSig001.setup()
        # G1ObsrSig002 = ObservationReward(observations, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50])
        # G1ObsrSig002.setup()
        # G1ObsrSig005 = ObservationReward(observations, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20])
        # G1ObsrSig005.setup()
        obsr_name_list = [
            # (G1ObsrSig001, "G1ObsrSig001"),
            # (G1ObsrSig002, "G1ObsrSig002"),
            # (G1ObsrSig005, "G1ObsrSig005"),
        ]
        S005unobsSig001 = UnobservedSD(observations, penalty=0.05, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100])
        S005unobsSig001.setup()
        # S01unobsSig001 = UnobservedSD(observations, penalty=0.05, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100])
        # S01unobsSig001.setup()
        # S01unobsSig002 = UnobservedSD(observations, penalty=0.1, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50])
        # S01unobsSig002.setup()
        # S03unobsSig001 = UnobservedSD(observations, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100])
        # S03unobsSig001.setup()
        # S03unobsSig002 = UnobservedSD(observations, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50])
        # S03unobsSig002.setup()
        unobs_name_list = [
            (S005unobsSig001, "S005unobsSig001"),
            # (S01unobsSig001, "S01unobsSig001"),
            # (S01unobsSig002, "S01unobsSig002"),
            # (S03unobsSig001, "S03unobsSig001"),
            # (S03unobsSig002, "S03unobsSig002"),
        ]
        Gerr1Sig001 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/100, (max(dm.smsz)-min(dm.smsz))/100], Gerr = 1.0)
        Gerr1Sig001.train()
        Gerr1Sig002 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/50, (max(dm.smsz)-min(dm.smsz))/50], Gerr = 1.0)
        Gerr1Sig002.train()
        # Gerr1Sig005 = GMM(dm.nnmodel, diag_sigma=[(max(dm.dtheta2)-min(dm.dtheta2))/20, (max(dm.smsz)-min(dm.smsz))/20], Gerr = 1.0)
        # Gerr1Sig005.train()
        gmm_name_list = [
            (Gerr1Sig001, "Gerr1_Sig001"),
            # (Gerr1Sig002, "Gerr1_Sig002"),
            # (Gerr1Sig005, "Gerr1_Sig005"),
        ]
        datotal = setup_datotal(dm, logdir)
        obsrcalcs = setup_obsr(dm, obsr_name_list, logdir)
        unobssds = setup_unobssd(dm, unobs_name_list, logdir)
        gmmpred = setup_gmmpred(dm, gmm_name_list, logdir)
        reward = setup_reward(dm, logdir)
        