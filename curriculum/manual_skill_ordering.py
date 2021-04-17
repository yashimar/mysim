from copy import deepcopy
from util import CreateExperimentsEvidenceFile
from tasks_domain.util import SetupDPL, CreateDPLLog
from tasks_domain import pouring as td
from tsim.dpl_cmn import *
from core_tool import *
SmartImportReload('tsim.dpl_cmn')


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
    default_space_defs = deepcopy(domain.SpaceDefs)
    l.dpl, fp = SetupDPL(ct, l, domain)

    count = 0
    is_ready = True
    while count <= l.num_episodes:
        if (len(l.tasks) >= 1) & is_ready:
            task = l.tasks[0]
            l.dpl.d.SpaceDefs.update(task.skill_params_def)
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
            is_ready = True

    fp.close()
    l = None


def Run(ct, *args):
    l = TContainer(debug=True)

    ############################################################################
    # Specify save directory
    ############################################################################
    suff = "ketchup_0055/third"+"/"
    l.logdir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
        + "curriculum/manual_skill_ordering"+"/"+suff

    ############################################################################
    # Specify src directory
    ############################################################################
    # l.src_core = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/' \
    #         + "bottomup/learn4/std_pour/ketchup/random/graphModel/modifiedStdPour/first"+"/"
    l.src_core = ""

    ############################################################################
    # Modify ConfigCallback
    ############################################################################
    l.config_callback = td.ConfigCallback
    l.rcv_size = "static"
    l.mtr_smsz = "custom"
    l.custom_mtr = "ketchup"
    l.custom_smsz = 0.055

    ############################################################################
    # Modify learning config
    ############################################################################
    l.num_episodes = 50
    l.interactive = False
    l.not_learn = False
    l.planning_node = ["n0"]
    l.opt_conf = {
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

    ############################################################################
    # Create Tasks
    ############################################################################
    SP = TCompSpaceDef

    t0 = Task(name="t0")  # Optimize p_pour_trg
    t0.skill_params_def = {
        'gh_ratio': SP('state', 1, min=[0.0], max=[1.0]),
        'dtheta1': SP('state', 1, min=[0.01], max=[0.02]),
        'dtheta2': SP('state', 1, min=[0.002], max=[0.02]),
        'shake_spd': SP('state', 1, min=[0.5], max=[1.2]),
        'shake_range': SP('state', 1, min=[0.05], max=[0.12]),
        'shake_angle': SP('state', 1, min=[-0.5*math.pi], max=[0.5*math.pi]),
    }
    t0.terminal_condition = lambda count: TerminalCheck(count, 9)

    t1 = Task(name="t1")  # Optimize p_pour_trg, shake_range
    t1.skill_params_def = {
        'gh_ratio': SP('state', 1, min=[0.0], max=[1.0]),
        'dtheta1': SP('state', 1, min=[0.01], max=[0.02]),
        'dtheta2': SP('state', 1, min=[0.002], max=[0.02]),
        'shake_spd': SP('state', 1, min=[0.5], max=[1.2]),
        'shake_angle': SP('state', 1, min=[-0.5*math.pi], max=[0.5*math.pi]),
    }
    t1.terminal_condition = lambda count: TerminalCheck(count, 5)

    t2 = Task(name="t2")  # Optimize p_pour_trg, shake_angle
    t2.skill_params_def = {
        'gh_ratio': SP('state', 1, min=[0.0], max=[1.0]),
        'dtheta1': SP('state', 1, min=[0.01], max=[0.02]),
        'dtheta2': SP('state', 1, min=[0.002], max=[0.02]),
        'shake_spd': SP('state', 1, min=[0.5], max=[1.2]),
        'shake_range': SP('state', 1, min=[0.05], max=[0.12]),
    }
    t2.terminal_condition = lambda count: TerminalCheck(count, 5)

    t3 = Task(name="t3")  # Optimize p_pour_trg, shake_range, shake_angle
    t3.skill_params_def = {
        'gh_ratio': SP('state', 1, min=[0.0], max=[1.0]),
        'dtheta1': SP('state', 1, min=[0.01], max=[0.02]),
        'dtheta2': SP('state', 1, min=[0.002], max=[0.02]),
        'shake_spd': SP('state', 1, min=[0.5], max=[1.2]),
    }
    t3.terminal_condition = lambda count: TerminalCheck(count, 5)

    t4 = Task(name="t4")  # Optimize p_pour_trg, shake_range, shake_angle, shake_spd
    t4.skill_params_def = {
        'gh_ratio': SP('state', 1, min=[0.0], max=[1.0]),
        'dtheta1': SP('state', 1, min=[0.01], max=[0.02]),
        'dtheta2': SP('state', 1, min=[0.002], max=[0.02]),
    }
    t4.terminal_condition = lambda count: TerminalCheck(count, 5)

    t5 = Task(name="t5")  # Optimize p_pour_trg, shake_range, shake_angle, shake_spd, gh_ratio
    t5.skill_params_def = {
        'dtheta1': SP('state', 1, min=[0.01], max=[0.02]),
        'dtheta2': SP('state', 1, min=[0.002], max=[0.02]),
    }
    t5.terminal_condition = lambda count: TerminalCheck(count, 5)

    l.tasks = [t0, t1, t2, t3, t4, t5]

    if True:
        ExecuteLearning(ct, l)
