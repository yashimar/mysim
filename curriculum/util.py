def CreateExperimentsEvidenceFile(l):
    l.logdir = l.logdir + "/" + l.suff
    print 'Copying', PycToPy(__file__), 'to', PycToPy(l.logdir+os.path.basename(__file__))
    CopyFile(PycToPy(__file__), PycToPy(l.logdir+os.path.basename(__file__)))
    if os.path.exists(l.logdir+"config_log.yaml") == False and os.path.exists(l.logdir+"dpl_est.dat") == False:
        if src_core != "":
            CopyFile(l.src_core+"config_log.yaml", l.logdir+"config_log.yaml")
            Print("Copying", l.src_core+"config_log.yaml", "to", l.logdir+"config_log.yaml")
            CopyFile(l.src_core+"dpl_est.dat", l.logdir+"dpl_est.dat")
            Print("Copying", l.src_core+"dpl_est.dat", "to", l.logdir+"dpl_est.dat")
        else:
            os.mknod(l.logdir+"config_log.yaml")
            os.mknod(l.logdir+"dpl_est.dat")
    else:
        pass