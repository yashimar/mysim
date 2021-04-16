def CreatePredictionLog(l, key, xs, ys):
    MM = l.dpl.MM
    In, Out, F = MM.Models[key]
    x_in, cov_in, dims_in = SerializeXSSA(MM.SpaceDefs, xs, In)
    x_out, cov_out, dims_out = SerializeXSSA(MM.SpaceDefs, ys, Out)
    pred = F.Predict(x_in, with_var=True)
    log = (key, {"input": x_in, "true_output": x_out, "prediction": {"X": pred.Y.tolist(), "Cov": pred.Var.tolist()}})
    l.pred_true_log.append(log)


def CreateDPL(ct, l, domain):
    if 'log_dpl' in ct.__dict__ and (CPrint(1, 'Restart from existing DPL?'), AskYesNo())[1]:
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

    return dpl, is_restarted


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
        yaml.dump({count-1: {key: data for (key, data) in l.pred_true_log}}, f, default_flow_style=False)
