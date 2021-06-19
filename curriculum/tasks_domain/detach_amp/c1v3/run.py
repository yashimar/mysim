from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from ....util import CreateExperimentsEvidenceFile
from ..util import SetupDPL, CreateDPLLog, check_or_create_dir
import task_domain as td
Rmodel = td.Rmodel
from copy import deepcopy
import random


ROOT_PATH = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'


def Help():
    pass


class Task:
    def __init__(self, name, group_id, group_task_id):
        self.name = "[{}]".format(name)
        self.group_id = group_id
        self.group_task_id = group_task_id
        self.skill_params_def = {}
        self.config_callback = lambda: None
        self.reward_callback = lambda: None
        self.pour_skill = ""
        self.border_return = -1.
        self.n_consider = 5
        self.return_log = []
        self.n_least_episode = 10


    def TerminalCheck(self):
        if len(self.return_log) < self.n_least_episode:
            return False
        elif all(r>self.border_return for r in self.return_log[-self.n_consider:]):
            return True
        else:
            return False
        
    
    def Update(self, l):
        self.return_log.append(l.dpl.DB.Entry[-1].R)
        
        
    def DumpResult(self, l):
        fp = open(l.logdir+'result.csv', "a")
        values = [self.group_id, self.group_task_id, len(self.return_log)] + [l.dpl.DB.Entry[-1].R] + [l.dpl.Value(tree) for tree in l.node_best_tree]
        idx = len(l.dpl.DB.Entry)-1
        fp.write('%i,%s\n' % (idx, ','.join(map(str, values))))
        fp.close()
        

def SetupSubtaskDPL(ct, l, td, group_id):
    l.logdir = l.logdir_base + "g{}/".format(group_id)
    if group_id == 0:
        l.db_src, l.opt_conf['model_dir'], l.opt_conf['db_src'] = "", "", ""
    else:
        l.db_src = l.logdir_base + "g{}".format(group_id-1)
        l.opt_conf['model_dir'] = "{}/models/".format(l.db_src)
        l.opt_conf['db_src'] = "{}/database.yaml".format(l.db_src)
    check_or_create_dir(l.logdir)
    CreateExperimentsEvidenceFile(l, __file__)
    
    domain = td.Domain()
    domain.SpaceDefs.update(l.skill_params_def)
    l.dpl, fp = SetupDPL(ct, l, domain, do_new_create = True)
    l.default_config_callback()
    l.default_reward_callback()
    l.pour_skill = ""
    return fp


def ExecuteLearning(ct, l):
    group_id = l.init_group_id
    max_group_id = max([task.group_id for task in l.tasks])
    done_all_subtask = False
    while not done_all_subtask:
        fp = SetupSubtaskDPL(ct, l, td, group_id)
        tasks = dict([(task.group_task_id, task) for task in l.tasks if task.group_id == group_id])
        done_tmp_subtask, count = False, 0
        while not done_tmp_subtask:
            task = tasks[random.choice(tasks.keys())]   # random.choice(tasks) has bug after done tasks.pop(...)
            l.dpl.d.SpaceDefs.update(task.skill_params_def)
            task.config_callback()
            task.reward_callback()
            l.pour_skill = task.pour_skill
            
            CPrint(2, '========== Start %4i ==========' % count)
            CPrint(3, task.name)
            CPrint(3, "[Debug] group_task_id: {}, return_log: {}".format(task.group_task_id, task.return_log))
            td.Execute(ct, l)
            task.Update(l)
            task.DumpResult(l)
            if task.TerminalCheck():
                tasks.pop(task.group_task_id)
            
            fp.write(l.dpl.DB.DumpOneYAML())
            fp.flush()
            CPrint(1, count, l.dpl.DB.DumpOne())
            CreateDPLLog(l, count)
            CPrint(2, '========== End %4i ==========' % count)
                        
            if len(tasks) == 0:
                fp.close()
                done_tmp_subtask = True
                group_id += 1
                if group_id > max_group_id:
                    done_all_subtask = True
            else:
                count += 1
        
    l = None


def Run(ct, *args):
    l = TContainer(debug=True)

    ############################################################################
    # Specify save directory
    ############################################################################
    t_index = 1
    suff = "t{}/".format(str(t_index))
    l.logdir_base = ROOT_PATH + "curriculum5/c1v3"+"/"+suff
    
    ##########################################################
    ### Specify start subtask group id
    ##########################################################
    l.init_group_id = 0

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
        "Rdatotal_gentle": [['da_trg', 'da_total'], [REWARD_KEY], TLocalQuad(2,lambda y: 0)],
        "Rdapour_gentle": [['da_trg', 'da_pour'], [REWARD_KEY], TLocalQuad(2,lambda y: 0)],
        "Rdaspill": [['da_spill2'], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
        "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: 0)],
    })

    ############################################################################
    # Modify learning config
    ############################################################################
    l.interactive = False
    l.not_learn = False
    l.planning_node = ["n0"]
    l.opt_conf = {
        'model_dir': "",    # Modify for each subtask
        'model_dir_persistent': False,  # If False, models are saved in l.logdir, i.e. different one from 'model_dir'
        'db_src': "",       # Modify for each subtask
        'config': {},  # Config of the simulator
        'dpl_options': {
            'opt_log_name': '{base}seq/opt-{i:04d}-{e:03d}-{n}-{v:03d}.dat',  # '{base}seq/opt-{i:04d}-{e:03d}-{n}-{v:03d}.dat' or None
            "ddp_sol": {
                # 'ptree_num': 60, #default auto
                'db_init_ratio': 0.0, #default 0.5
                # 'num_finished': 60,  #default 20
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
    nn_options_base = {
        "batchsize": 10,  # default 10
        "num_max_update": 5000,  # default 5000
        'num_check_stop': 50,  # default 50
        'loss_stddev_stop': 2e-5,  # default 1e-3
        'AdaDelta_rho': 0.9,  # default 0.9
        # 'train_log_file': '{base}train/nn_log-{name}{code}.dat',
        # "train_batch_loss_log_file": '{base}train/nn_batch_loss_log-{name}{code}.dat',
    }
    l.nn_options = {
        'Fgrasp': nn_options_base,
        'Fmvtorcv_rcvmv': nn_options_base,
        'Fmvtorcv': nn_options_base,
        'Fmvtopour2': nn_options_base,
        'Ftip_amount': nn_options_base,
        'Ftip_flow': nn_options_base,
        'Fshake_amount': nn_options_base,
        'Fshake_flow': nn_options_base,
        'Famount': nn_options_base,
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
    
    ###########################
    ### Group_id: 0
    ###########################
    ### group_task_id: 0
    l.tasks.append(Task(name="initial sample(tip)", group_id=0, group_task_id=0))
    l.tasks[-1].border_return = -100.
    l.tasks[-1].pour_skill = "tip"
    l.tasks[-1].n_least_episode = 3
    
    ### group_task_id: 1
    l.tasks.append(Task(name="initial sample(shake)", group_id=0, group_task_id=1))
    l.tasks[-1].border_return = -100.
    l.tasks[-1].pour_skill = "shake"
    l.tasks[-1].n_least_episode = 3
    
    ###########################
    ### Group_id: 1
    ###########################
    #task_id: 0
    l.tasks.append(Task(name="mtr=(nobounce,ketchup), smsz=(0.03,0.08), Rdapour_gentle + Rdaspill", group_id=1, group_task_id=0))
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rdaspill": [['da_spill2'], [REWARD_KEY], Rmodel("Fdaspill")],
        "Rdapour_gentle": [['da_trg', 'da_pour'], [REWARD_KEY], Rmodel("Fdapour_gentle")],
    })
    l.tasks[-1].border_return = -100.
    l.tasks[-1].n_least_episode = 294

    ############################################################################
    # Execute
    ############################################################################
    if True:
        ExecuteLearning(ct, l)
