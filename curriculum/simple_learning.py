from core_tool import *
from tasks_domain import pouring as td
from tasks_domain.util import SetupDPL, CreateDPLLog
from util import CreateExperimentsEvidenceFile


def Help():
    pass


def ExecuteLearning(ct, l):
    print 'Copying',PycToPy(__file__),'to',PycToPy(l.logdir+os.path.basename(__file__))
    CopyFile(PycToPy(__file__),PycToPy(l.logdir+os.path.basename(__file__)))

    domain = td.Domain()
    l.dpl, fp = SetupDPL(ct, l, domain)

    for count in l.opt_conf['num_episodes']:
        CPrint(2, '========== Start %4i ==========' % count)
        pouring.Execute(ct, l)
        CPrint(2, '========== End %4i ==========' % count)

        fp.write(l.dpl.DB.DumpOneYAML())
        fp.flush()
        CPrint(1,count,l.dpl.DB.DumpOne())

        CreateDPLLog(l, count)

    fp.close()
    l= None


def Run(ct, *args):
    l = TContainer(debug=True)

    ############################################################################
    # Specify save directory
    ############################################################################
    l.logdir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
        + "curriculum/simple_learning"+"/"
    l.suff = "test"+"/"

    ############################################################################
    # Specify src directory
    ############################################################################
    # l.src_core = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
    #         + "bottomup/learn4/std_pour/nobounce/random/graphModel/modifiedStdPour/first"+"/"
    l.src_core = ""

    ############################################################################
    # Modify ConfigCallback
    ############################################################################
    l.config_callback = td.ConfigCallback
    l.custom_mtr = "natto"
    l.custom_smsz = 0.055

    ############################################################################
    # Modify learning config
    ############################################################################
    l.opt_conf = {
        'num_episodes': 3,
        'interactive': False,
        'not_learn': False,
        'rcv_size': 'static',  # 'static', 'random'
        'mtr_smsz': 'custom',  # 'fixed', 'fxvs1', 'random', 'viscous', custom
        "planning_node": ["n0"],  # "n0","n2a"
        'model_dir': l.src_core + "models/" if l.src_core != "" else "",
        'model_dir_persistent': False,  # If False, models are saved in l.logdir, i.e. different one from 'model_dir'
        'db_src': l.src_core + "database.yaml" if l.src_core != "" else "",
        'config': {},  # Config of the simulator
        'dpl_options': {
            'opt_log_name': '{base}seq/opt-{i:04d}-{e:03d}-{n}-{v:03d}.dat',  # '{base}seq/opt-{i:04d}-{e:03d}-{n}-{v:03d}.dat' or None
            "ddp_sol": {
                # 'ptree_num': 40, #default auto
                # 'db_init_ratio': 1.0, #default 0.5
                'db_init_R_min': -1.0,  # default -1.0
                'grad_max_bounce': 10,  # default 10
                'prob_update_best': 0.4,  # default 0.4
                'prob_update_rand': 0.3,  # default 0.3
                'max_total_iter': 2000,  # default 2000
                "grad_max_iter": 50,  # default 50
                'gd_alpha': 0.03  # default 0.03
            },
        },
    }

    ############################################################################
    # Modify NN training option
    ############################################################################
    l.nn_options = {
        "batchsize": 10,  # default 10
        "num_max_update": 5000,  # default 5000
        'num_check_stop': 50,  # default 50
        'loss_stddev_stop': 1e-3,  # default 1e-3
        'AdaDelta_rho': 0.9,  # default 0.9
        # 'train_log_file': '{base}train/nn_log-{name}{code}.dat',
        # "train_batch_loss_log_file": '{base}train/nn_batch_loss_log-{name}{code}.dat',
    }

    if True:
        ExecuteLearning(ct, l)