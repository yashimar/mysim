from ...util import CreateExperimentsEvidenceFile
from ..util import SetupDPL, CreateDPLLog
import task_domain as td
from core_tool import *

ROOT_PATH = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'


def Help():
    pass


def ExecuteLearning(ct, l):
    CreateExperimentsEvidenceFile(l, __file__)

    domain = td.Domain()
    domain.SpaceDefs.update(l.skill_params_def)
    l.dpl, fp = SetupDPL(ct, l, domain)

    for count in range(l.num_episodes):
        if count <= 2:
            l.pour_skill = "tip"
        elif count <= 5:
            l.pour_skill = "shake"
        else:
            l.pour_skill = ""
        CPrint(2, '========== Start %4i ==========' % count)
        td.Execute(ct, l)
        CPrint(2, '========== End %4i ==========' % count)

        fp.write(l.dpl.DB.DumpOneYAML())
        fp.flush()
        CPrint(1, count, l.dpl.DB.DumpOne())

        CreateDPLLog(l, count)

    fp.close()
    l = None


def Run(ct, *args):
    l = TContainer(debug=True)

    ############################################################################
    # Specify save directory
    ############################################################################
    suff = "curriculum_test/test/first"+"/"
    l.logdir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
                + "curriculum/pouring/scratch"+"/"+suff

    ############################################################################
    # Specify src directory
    ############################################################################
    # l.db_src = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
    #         + "bottomup/learn4/std_pour/ketchup/random/graphModel/modifiedStdPour/first"
    l.db_src = ""
    l.model_src = ROOT_PATH + "curriculum/flowing_out/scratch/curriculum_test/test/first"
    # l.model_src = ""

    ############################################################################
    # Modify ConfigCallback
    ############################################################################
    l.config_callback = td.ConfigCallback
    l.rcv_size = "static"
    l.mtr_smsz = "curriculum_test"

    ############################################################################
    # Modify learning config
    ############################################################################
    l.num_episodes = 7
    l.interactive = False
    l.not_learn = False
    l.planning_node = ["n0"]
    l.opt_conf = {
        'model_dir': l.model_src + "/models/" if l.model_src != "" else "",
        'model_dir_persistent': False,  # If False, models are saved in l.logdir, i.e. different one from 'model_dir'
        'db_src': l.db_src + "/database.yaml" if l.db_src != "" else "",
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

    ############################################################################
    # Modify Domain
    ############################################################################
    SP = TCompSpaceDef
    l.skill_params_def = {
        'dtheta1': SP('state', 1, min=[0.01], max=[0.02]),
        'gh_ratio': SP('state', 1, min=[0.0], max=[1.0]),
    }

    if True:
        ExecuteLearning(ct, l)
