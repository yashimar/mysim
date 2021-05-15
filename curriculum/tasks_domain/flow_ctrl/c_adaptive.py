from copy import deepcopy

from matplotlib.pyplot import title
from ...util import CreateExperimentsEvidenceFile
from ..util import SetupDPL, CreateDPLLog
import task_domain as td
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
        self.reward_callback = lambda: None
        self.custom_reward_model = {}
        self.terminal_condition = lambda: None
        self.pour_skill = ""


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
    l.default_config_callback()
    l.default_reward_callback()
    while count < l.num_episodes:
        if (len(l.tasks) >= 1) & is_ready:
            task = l.tasks[0]
            l.dpl.d.SpaceDefs.update(task.skill_params_def)
            task.config_callback()
            task.reward_callback()
            l.pour_skill = task.pour_skill
            i = 0
            is_ready = False

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
            l.default_reward_callback()
            l.pour_skill = ""
            is_ready = True

    fp.close()
    l = None


def Run(ct, *args):
    l = TContainer(debug=True)

    ############################################################################
    # Specify save directory
    ############################################################################
    t_index = 1
    suff = "curriculum_test/t"+str(t_index)+"/c8_large_nobounce_tip_5_5_5_5"+"/"
    l.logdir = ROOT_PATH + "curriculum/flow_ctrl/c_adaptive"+"/"+suff

    ############################################################################
    # Specify src directory
    ############################################################################
    # l.db_src = ROOT_PATH + "curriculum/outflow3/c5/curriculum_test/t"+str(t_index)+"/first80"
    l.db_src = ""
    l.model_src = ROOT_PATH + "curriculum/flow_ctrl/c_adaptive/curriculum_test/t"+str(t_index)+"/c8_large_nobounce_tip_5_5_5"
    # l.model_src = ""

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
    # Modify reward function
    ############################################################################
    def Delta1(dim, s):
        assert(abs(s-int(s)) < 1.0e-6)
        p = [0.0]*dim
        p[int(s)] = 1.0
        return p
    def update_model(new_models):
        l.dpl.d.Models.update(new_models)
    l.default_reward_callback = lambda: update_model({
        "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
        "Rdapour_gentle": [['da_trg', 'da_pour'], [REWARD_KEY], TLocalQuad(2,lambda y: 0)],
    })

    ############################################################################
    # Modify learning config
    ############################################################################
    l.num_episodes = 5
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
                'ptree_num': 50, #default auto
                'db_init_ratio': 0.0, #default 0.5
                'num_finished': 60,  #default 20
                'num_proc': 12,  #default 12
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
    
    # # 0
    # l.tasks.append(Task(name="init sample: tip"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "curriculum_test", "", "")
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 3)
    # l.tasks[-1].pour_skill = "tip"

    # l.tasks.append(Task(name="init sample: shake"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "curriculum_test", "", "")
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 3)
    # l.tasks[-1].pour_skill = "shake"
    
    # l.tasks.append(Task(name="tip concentration"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce", "ketchup"), (0.03, 0.08))
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 22)
    # l.tasks[-1].reward_callback = lambda: update_model({
    #     "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: -100.0 if y[0]!=0 else 0)],
    #     "Rdaspill": [["da_spill2"], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    # })
    # l.tasks[-1].skill_params_def = {
    #     'p_pour_trg': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),
    # }
    
    # l.tasks.append(Task(name="shake concentration"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce", "ketchup"), (0.03, 0.08))
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 22)
    # l.tasks[-1].reward_callback = lambda: update_model({
    #     "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: -100.0 if y[0]!=1 else 0)],
    #     "Rdaspill": [["da_spill2"], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    # })
    # l.tasks[-1].skill_params_def = {
    #     'p_pour_trg': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),
    # }
    
    # #1
    # l.tasks.append(Task(name="small"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce", "ketchup"), (0.07, 0.08))
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 5)
    # l.tasks[-1].reward_callback = lambda: update_model({
    #     "Rdaspill": [["da_spill2"], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    # })
    # l.tasks[-1].skill_params_def = {
    #     'p_pour_trg': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),
    # }
    
    # #2
    # l.tasks.append(Task(name="small nobounce tip dtheta2"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce",), (0.077, 0.079))
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 3)
    # l.tasks[-1].reward_callback = lambda: update_model({
    #     "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: -100.0 if y[0]!=0 else 0)],
    #     "Rdaspill": [["da_spill2"], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    # })
    # l.tasks[-1].skill_params_def = {
    #     'p_pour_trg': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),
    #     'dtheta2': SP('action', 1, min=[0.018], max=[0.02]),
    # }
    
    # #3
    # l.tasks.append(Task(name="small"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce", "ketchup"), (0.07, 0.08))
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 5)
    # l.tasks[-1].reward_callback = lambda: update_model({
    #     "Rdaspill": [["da_spill2"], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    # })
    # l.tasks[-1].skill_params_def = {
    #     'p_pour_trg': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),
    # }
    
    # #4
    # l.tasks.append(Task(name="small ketchup tip"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("ketchup",), (0.078, 0.08))
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 5)
    # l.tasks[-1].reward_callback = lambda: update_model({
    #     "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: -100.0 if y[0]!=0 else 0)],
    #     "Rdaspill": [["da_spill2"], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    # })
    # l.tasks[-1].skill_params_def = {
    #     'p_pour_trg': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),
    #     'dtheta2': SP('action', 1, min=[0.002], max=[0.004]),
    # }
    
    # #5
    # l.tasks.append(Task(name="middle nobounce"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce", ), (0.05, 0.07))
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 5)
    # l.tasks[-1].reward_callback = lambda: update_model({
    #     "Rdaspill": [["da_spill2"], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    # })
    # l.tasks[-1].skill_params_def = {
    #     'p_pour_trg': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),
    # }
    
    # #6
    # l.tasks.append(Task(name="middle ketchup"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("ketchup", ), (0.05, 0.07))
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 5)
    # l.tasks[-1].reward_callback = lambda: update_model({
    #     "Rdaspill": [["da_spill2"], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    # })
    # l.tasks[-1].skill_params_def = {
    #     'p_pour_trg': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),
    # }
    
    # #7
    # l.tasks.append(Task(name="large nobounce"))
    # l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce", ), (0.03, 0.05))
    # l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 5)
    # l.tasks[-1].reward_callback = lambda: update_model({
    #     "Rdaspill": [["da_spill2"], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    # })
    # l.tasks[-1].skill_params_def = {
    #     'p_pour_trg': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),
    # }
    
    #8
    l.tasks.append(Task(name="large nobounce tip"))
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce", ), (0.03, 0.05))
    l.tasks[-1].terminal_condition = lambda count: TerminalCheck(count, 5)
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: -100.0 if y[0]!=0 else 0)],
        "Rdaspill": [["da_spill2"], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    })
    l.tasks[-1].skill_params_def = {
        'p_pour_trg': SP('state', 2, min=[0.2, 0.1], max=[1.2, 0.7]),
    }

    ############################################################################
    # Execute
    ############################################################################
    if True:
        ExecuteLearning(ct, l)