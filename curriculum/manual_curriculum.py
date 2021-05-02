from copy import deepcopy

from matplotlib.pyplot import title
from util import CreateExperimentsEvidenceFile
from tasks_domain.util import SetupDPL, CreateDPLLog
from tasks_domain import pouring as td
from tsim.dpl_cmn import *
from core_tool import *
SmartImportReload('tsim.dpl_cmn')

ROOT_PATH = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'


def Help():
    pass


class Task:
    def __init__(self, name):
        self.name = name
        self.skill_params_def = {}
        self.config_callback = lambda: None
        self.reward_function = {}
        self.terminal_condition = lambda: None


def TerminalCheck(count, max_count):
    if count >= max_count:
        return True
    else:
        return False


def ExecuteLearning(ct, l):
    CreateExperimentsEvidenceFile(l, __file__)

    domain = td.Domain()
    domain.SpaceDefs.update(l.skill_params_def)
    default_space_defs = deepcopy(domain.SpaceDefs)
    l.dpl, fp = SetupDPL(ct, l, domain)

    count = 0
    is_ready = True
    while count < l.num_episodes:
        if (len(l.tasks) >= 1) & is_ready:
            task = l.tasks[0]
            l.dpl.d.SpaceDefs.update(task.skill_params_def)
            task.config_callback()
            i = 0
            is_ready = False

        if count <= 2:
            l.pour_skill = "std_pour"
        elif count <= 5:
            l.pour_skill = "shake_A"
        else:
            l.pour_skill = ""

        CPrint(2, '========== Start %4i ==========' % count)
        td.Execute(ct, l)
        CPrint(2, '========== End %4i ==========' % count)

        fp.write(l.dpl.DB.DumpOneYAML())
        fp.flush()
        CPrint(1, count, l.dpl.DB.DumpOne())
        CreateDPLLog(l, count)

        count += 1
        if is_ready == False:
            i += 1
        if task.terminal_condition(i) & (is_ready == False):
            l.tasks.pop(0)
            l.dpl.d.SpaceDefs.update(default_space_defs)
            l.default_config_callback()
            is_ready = True

    fp.close()
    l = None


def Run(ct, *args):
    l = TContainer(debug=True)

    ############################################################################
    # Specify save directory
    ############################################################################
    suff = "t6/first"+"/"
    l.logdir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
                + "curriculum/manual_init_state_ordering"+"/"+suff

    ############################################################################
    # Specify src directory
    ############################################################################
    # l.db_src = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
    #         + "bottomup/learn4/std_pour/ketchup/random/graphModel/modifiedStdPour/first"+"/"
    l.db_src = ""
    # l.model_src = ""
    l.model_src = ROOT_PATH + "curriculum/pretrain/Fmvtopour2"

    ############################################################################
    # Modify ConfigCallback
    ############################################################################
    l.config_callback = td.ConfigCallback

    def custom_config_callback(rcv_size, mtr_smsz, custom_mtr, custom_smsz):
        l.rcv_size = rcv_size
        l.mtr_smsz = mtr_smsz
        l.custom_mtr = custom_mtr
        l.custom_smsz = custom_smsz
    l.default_config_callback = lambda: custom_config_callback("static", "curriculum_test", "", "")

    ############################################################################
    # Modify learning config
    ############################################################################
    l.num_episodes = 120
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

    ############################################################################
    # Create Tasks
    ############################################################################
    SP = TCompSpaceDef
    l.tasks = []

    # 1
    l.tasks.append(Task(name="nobounce, 0.03"))
    l.tasks[-1].skill_params_def = {}
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", "nobounce", 0.03)
    l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 20)

    # 2
    l.tasks.append(Task(name="(nobounce, 0.055), (nobounce, 0.08), (ketchup, 0.03), (ketchup, 0.055)"))
    l.tasks[-1].skill_params_def = {}
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "middle_ease_of_flow", "", "")
    l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 40)

    # 3
    l.tasks.append(Task(name="ketchup, 0.08"))
    l.tasks[-1].skill_params_def = {}
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", "ketchup", 0.08)
    l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 20)

    ############################################################################
    # Execute
    ############################################################################
    if True:
        ExecuteLearning(ct, l)
