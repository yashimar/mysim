import yaml
import joblib
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
import os


def check_or_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def CurrentPredict(l, key, xs):
    MM = l.dpl.MM
    In, Out, F = MM.Models[key]
    x_in, cov_in, dims_in = SerializeXSSA(MM.SpaceDefs, xs, In)
    pred = F.Predict(x_in, with_var=True)
    pred_mean, pred_var = pred.Y.tolist(), pred.Var.tolist()
    return pred_mean, np.diag(pred_var)


def CreatePredictionLog(l, key, xs, ys):
    MM = l.dpl.MM
    In, Out, F = MM.Models[key]
    x_in, cov_in, dims_in = SerializeXSSA(MM.SpaceDefs, xs, In)
    x_out, cov_out, dims_out = SerializeXSSA(MM.SpaceDefs, ys, Out)
    pred = F.Predict(x_in, with_var=True)
    log = (key, {"input": x_in, "true_output": x_out, "prediction": {"X": pred.Y.tolist(), "Cov": pred.Var.tolist()}})
    l.pred_true_log.append(log)


def SetupDPL(ct, l, domain, do_new_create = False):
    if not do_new_create and 'log_dpl' in ct.__dict__ and (CPrint(1, 'Restart from existing DPL?'), AskYesNo())[1]:
        dpl = ct.log_dpl
        is_restarted = True
    else:
        mm_options = {
            # 'type': l.type,
            'base_dir': l.logdir+'models/',
        }
        mm = TModelManager(domain.SpaceDefs, domain.Models)
        mm.Load({'options': mm_options})
        if l.opt_conf['model_dir'] not in ('', None):
            if os.path.exists(l.opt_conf['model_dir']+'model_mngr.yaml'):
                mm.Load(LoadYAML(l.opt_conf['model_dir']+'model_mngr.yaml'), l.opt_conf['model_dir'])
            else:
                raise(Exception("Not exists : "+l.opt_conf['model_dir']+'model_mngr.yaml'))
            if l.opt_conf['model_dir_persistent']:
                mm.Options['base_dir'] = l.opt_conf['model_dir']
            else:
                mm.Options['base_dir'] = mm_options['base_dir']
        db = TGraphEpisodeDB()
        if l.opt_conf['db_src'] not in ('', None):
            db.Load(LoadYAML(l.opt_conf['db_src']))

        dpl = TGraphDynPlanLearn(domain, db, mm)
        is_restarted = False

    dpl_options = {
        'base_dir': l.logdir,
    }
    InsertDict(dpl_options, l.opt_conf['dpl_options'])
    dpl.Load({'options': dpl_options})

    if not is_restarted:
        dpl.MM.Init()
        dpl.Init()

    ct.log_dpl = dpl

    check_or_create_dir(l.logdir)
    print 'Copying', PycToPy(__file__), 'to', PycToPy(l.logdir+os.path.basename(__file__))
    CopyFile(PycToPy(__file__), PycToPy(l.logdir+os.path.basename(__file__)))

    if is_restarted:
        fp = OpenW(l.logdir+'dpl_log.dat', 'a', l.interactive)
        if len(dpl.DB.Entry) > 0:
            for i in range(len(dpl.DB.Entry)):
                fp.write(dpl.DB.DumpOneYAML(i))
            fp.flush()
    elif os.path.exists(l.logdir+"dpl_log.dat"):
        raise(Exception("Already log file exists. Change l.logdir."))
    else:
        fp = OpenW(l.logdir+'dpl_log.dat', 'w', l.interactive)
        if len(dpl.DB.Entry) > 0:
            for i in range(len(dpl.DB.Entry)):
                fp.write(dpl.DB.DumpOneYAML(i))
            fp.flush()

    return dpl, fp


def CreateDPLLog(l, count):
    SaveYAML(l.dpl.MM.Save(l.dpl.MM.Options['base_dir']), l.dpl.MM.Options['base_dir']+'model_mngr.yaml')
    SaveYAML(l.dpl.DB.Save(), l.logdir+'database.yaml')
    SaveYAML(l.dpl.Save(), l.logdir+'dpl.yaml')

    config = {key: getattr(l.config, key) for key in l.config.__slots__}
    l.config_log = [config]
    # l.config_log.append(config)
    # SaveYAML(l.config_log, l.logdir+'config_log.yaml', interactive=False)

    # if l.restarting==True or count>1: w_mode = "a"
    # else: w_mode = "w"
    w_mode = "a"
    OpenW(l.logdir+'config_log.yaml', mode=w_mode, interactive=False).write(yamldump(ToStdType(l.config_log, lambda y: y), Dumper=YDumper))

    fp = open(l.logdir+'dpl_est.dat', w_mode)
    values = [l.dpl.DB.Entry[-1].R] + [l.dpl.Value(tree) for tree in l.node_best_tree]
    idx = len(l.dpl.DB.Entry)-1
    fp.write('%i %s\n' % (idx, ' '.join(map(str, values))))
    fp.close()
    if w_mode == "a":
        CPrint(1, 'Generated:', l.logdir+'dpl_est.dat')
    else:
        CPrint(1, 'Added:', l.logdir+'dpl_est.dat')

    if not os.path.exists(l.logdir+"best_est_trees"):
        os.mkdir(l.logdir+"best_est_trees")
    for i, tree in enumerate(l.node_best_tree):
        if i == 0:
            joblib.dump(tree, l.logdir+"best_est_trees/"+"ep"+str(len(l.dpl.DB.Entry)-1)+"_n0.jb")
        else:
            joblib.dump(tree, l.logdir+"best_est_trees/"+"ep"+str(len(l.dpl.DB.Entry)-1)+"_n2a_"+str(i)+".jb")

    with open(l.logdir+'pred_true_log.yaml', w_mode) as f:
        yaml.dump({count: {key: data for (key, data) in l.pred_true_log}}, f, default_flow_style=False)


def ModelManager(domain, model_path):
    mm = TModelManager(domain.SpaceDefs, domain.Models)
    mm.Load(LoadYAML(model_path+'/models/model_mngr.yaml'), model_path+"/models/")
    mm.Init()

    return mm


def Rmodel(model_name):
    modeldir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'\
                + 'reward_model'+"/"
    FRwd = TNNRegression()
    prefix = modeldir+'p1_model/'+model_name
    FRwd.Load(LoadYAML(prefix+'.yaml'), prefix+"/")
    FRwd.Init()

    return FRwd