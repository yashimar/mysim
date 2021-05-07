from core_tool import *

def CreateExperimentsEvidenceFile(l, file):
    print 'Copying',PycToPy(file),'to',PycToPy(l.logdir+os.path.basename(file))
    CopyFile(PycToPy(file),PycToPy(l.logdir+os.path.basename(file)))
    if not os.path.isdir(l.logdir):
        os.makedirs(l.logdir)
    if os.path.exists(l.logdir+"/config_log.yaml") == False and os.path.exists(l.logdir+"/dpl_est.dat") == False:
        if l.db_src != "":
            CopyFile(l.db_src+"/config_log.yaml", l.logdir+"/config_log.yaml")
            Print("Copying", l.db_src+"/config_log.yaml", "to", l.logdir+"/config_log.yaml")
            CopyFile(l.db_src+"/dpl_est.dat", l.logdir+"/dpl_est.dat")
            Print("Copying", l.db_src+"/dpl_est.dat", "to", l.logdir+"/dpl_est.dat")
        else:
            os.mknod(l.logdir+"/config_log.yaml")
            os.mknod(l.logdir+"/dpl_est.dat")
    else:
        pass