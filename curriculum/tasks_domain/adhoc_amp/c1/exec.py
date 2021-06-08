from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from ....util import CreateExperimentsEvidenceFile
from ..util import SetupDPL, CreateDPLLog
import task_domain as td
Rmodel = td.Rmodel
from copy import deepcopy


ROOT_PATH = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'


def Help():
    pass


class Task:
    def __init__(self, name, group_id):
        self.name = "[{}]".format(name)
        self.group_id = group_id
        self.skill_params_def = {}
        self.config_callback = lambda: None
        self.reward_callback = lambda: None
        self.pour_skill = ""
        self.border_return = -1.
        self.n_consider = 3
        self.return_log = []
        self.n_least_episode = 5


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
        fp = open(l.logdir+'subatask_result.dat', "a")
        values = [self.group_id, self.name, len(self.return_log)] + [l.dpl.DB.Entry[-1].R] + [l.dpl.Value(tree) for tree in l.node_best_tree]
        idx = len(l.dpl.DB.Entry)-1
        fp.write('%i %s\n' % (idx, ' '.join(map(str, values))))
        fp.close()


def ExecuteLearning(ct, l):
    CreateExperimentsEvidenceFile(l, __file__)

    domain = td.Domain()
    domain.SpaceDefs.update(l.skill_params_def)
    default_space_defs = deepcopy(domain.SpaceDefs)
    l.dpl, fp = SetupDPL(ct, l, domain)

    count = 0
    group_id = 0
    is_ready = True
    l.default_config_callback()
    l.default_reward_callback()
    while count < l.num_episodes:
        if is_ready:
            tasks = [task for task in l.tasks if task.group_id == group_id]
            is_ready = False
        group_task_id = RandI(len(tasks))
        task = tasks[group_task_id]
        l.dpl.d.SpaceDefs.update(task.skill_params_def)
        task.config_callback()
        task.reward_callback()
        l.pour_skill = task.pour_skill    

        CPrint(2, '========== Start %4i ==========' % count)
        td.Execute(ct, l)
        task.Update(l)
        task.DumpResult(l)
        CPrint(3, "Debug: group_task_id: {}, return_log: {}".format(group_task_id, task.return_log))
        if task.TerminalCheck():
            tasks.pop(group_task_id)
        
        fp.write(l.dpl.DB.DumpOneYAML())
        fp.flush()
        CPrint(1, count, l.dpl.DB.DumpOne())
        CreateDPLLog(l, count)
        CPrint(2, '========== End %4i ==========' % count)

        count += 1
        if len(tasks) == 0:
            l.dpl.d.SpaceDefs.update(default_space_defs)
            l.default_config_callback()
            l.default_reward_callback()
            l.pour_skill = ""
            group_id += 1
            is_ready = True

    fp.close()
    l = None


def Run(ct, *args):
    l = TContainer(debug=True)

    ############################################################################
    # Specify save directory
    ############################################################################
    t_index = 1
    suff = "curriculum_test/t"+str(t_index)+"/first50"+"/"
    l.logdir = ROOT_PATH + "curriculum4/c1"+"/"+suff

    ############################################################################
    # Specify src directory
    ############################################################################
    # l.db_src = ROOT_PATH + "curriculum/outflow3/c5/curriculum_test/t"+str(t_index)+"/first80"
    # l.model_src = ROOT_PATH + "curriculum/flow_ctrl/c_adaptive/curriculum_test/t"+str(t_index)+"/c8_large_nobounce_tip_5_5_5"
    l.db_src = ""
    l.model_src = ""

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
    l.num_episodes = 50
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
                'ptree_num': 60, #default auto
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
    
    ###########################
    ### Group_id: 0
    ###########################
    l.tasks.append(Task(name="mtr=nobounce, smsz=0.04+/-0.002, initial sample(tip)", group_id=0))
    l.tasks[-1].skill_params_def = {
        'p_pour_trg': SP('state', 2, min=[0.33, 0.4], max=[0.6, 0.5]),
    }
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce",), (0.038, 0.042))
    l.tasks[-1].border_return = -100.
    l.tasks[-1].pour_skill = "tip"
    l.tasks[-1].n_least_episode = 3
    
    ###########################
    ### Group_id: 1
    ###########################
    l.tasks.append(Task(name="mtr=ketchup, smsz=0.07+/-0.002, initial sample(shake)", group_id=1))
    l.tasks[-1].skill_params_def = {
        'p_pour_trg': SP('state', 2, min=[0.33, 0.4], max=[0.6, 0.5]),
    }
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("ketchup",), (0.068, 0.072))
    l.tasks[-1].border_return = -100.
    l.tasks[-1].pour_skill = "shake"
    l.tasks[-1].n_least_episode = 3

    ###########################
    ### Group_id: 2
    ###########################
    l.tasks.append(Task(name="mtr=nobounce, smsz=0.04+/-0.002, Rdatotal_gentle + Rskill(tip)", group_id=2))
    l.tasks[-1].skill_params_def = {
        'p_pour_trg': SP('state', 2, min=[0.33, 0.4], max=[0.6, 0.5]),
    }
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce",), (0.038, 0.042))
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rdatotal_gentle": [['da_trg', 'da_total'], [REWARD_KEY], Rmodel("Fdatotal_gentle")],
        "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: -100.0 if y[0]!=0 else 0)],
    })
    
    ###########################
    ### Group_id: 3
    ###########################
    l.tasks.append(Task(name="mtr=ketchup, smsz=0.07+/-0.002, Rdatotal_gentle + Rskill(shake)", group_id=3))
    l.tasks[-1].skill_params_def = {
        'p_pour_trg': SP('state', 2, min=[0.33, 0.4], max=[0.6, 0.5]),
    }
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("ketchup",), (0.068, 0.072))
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rdatotal_gentle": [['da_trg', 'da_total'], [REWARD_KEY], Rmodel("Fdatotal_gentle")],
        "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: -100.0 if y[0]!=1 else 0)],
    })
    
    ###########################
    ### Group_id: 4
    ###########################
    l.tasks.append(Task(name="mtr=nobounce, smsz=0.04+/-0.002, Rdaspill + Rskill(tip)", group_id=4))
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce",), (0.038, 0.042))
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rdaspill": [['da_spill2'], [REWARD_KEY], Rmodel("Fdaspill")],
        "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: -100.0 if y[0]!=0 else 0)],
    })
    
    ###########################
    ### Group_id: 5
    ###########################
    l.tasks.append(Task(name="mtr=ketchup, smsz=0.07+/-0.002, Rdaspill + Rskill(shake)", group_id=5))
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("ketchup",), (0.068, 0.072))
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rdaspill": [['da_spill2'], [REWARD_KEY], Rmodel("Fdaspill")],
        "Rskill": [['skill'], [REWARD_KEY], TLocalQuad(1,lambda y: -100.0 if y[0]!=1 else 0)],
    })
    
    ###########################
    ### Group_id: 6
    ###########################
    ### group_task_id: 0
    l.tasks.append(Task(name="mtr=nobounce, smsz=(0.03,0.055), Rdatotal_gentle + Rdaspill", group_id=6))
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce",), (0.03, 0.055))
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rdaspill": [['da_spill2'], [REWARD_KEY], Rmodel("Fdaspill")],
        "Rdatotal_gentle": [['da_trg', 'da_total'], [REWARD_KEY], Rmodel("Fdatotal_gentle")],
    })

    ### group_task_id: 1
    l.tasks.append(Task(name="mtr=ketchup, smsz=(0.055,0.08), Rdatotal_gentle + Rdaspill", group_id=6))
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("ketchup",), (0.055, 0.08))
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rdaspill": [['da_spill2'], [REWARD_KEY], Rmodel("Fdaspill")],
        "Rdatotal_gentle": [['da_trg', 'da_total'], [REWARD_KEY], Rmodel("Fdatotal_gentle")],
    })
    
    ###########################
    ### Group_id: 7
    ###########################
    ### group_task_id: 0
    l.tasks.append(Task(name="mtr=nobounce, smsz=(0.055,0.08), Rdatotal_gentle + Rdaspill", group_id=7))
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("nobounce",), (0.055, 0.08))
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rdaspill": [['da_spill2'], [REWARD_KEY], Rmodel("Fdaspill")],
        "Rdatotal_gentle": [['da_trg', 'da_total'], [REWARD_KEY], Rmodel("Fdatotal_gentle")],
    })

    ### group_task_id: 1
    l.tasks.append(Task(name="mtr=ketchup, smsz=(0.03,0.055), Rdatotal_gentle + Rdaspill", group_id=7))
    l.tasks[-1].config_callback = lambda: custom_config_callback("static",  "custom", ("ketchup",), (0.03, 0.055))
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rdaspill": [['da_spill2'], [REWARD_KEY], Rmodel("Fdaspill")],
        "Rdatotal_gentle": [['da_trg', 'da_total'], [REWARD_KEY], Rmodel("Fdatotal_gentle")],
    })
    
    ###########################
    ### Group_id: 8
    ###########################
    l.tasks.append(Task(name="mtr=(nobounce,ketchup), smsz=(0.03,0.08), Rdatotal_gentle + Rdaspill", group_id=8))
    l.tasks[-1].reward_callback = lambda: update_model({
        "Rdaspill": [['da_spill2'], [REWARD_KEY], Rmodel("Fdaspill")],
        "Rdatotal_gentle": [['da_trg', 'da_total'], [REWARD_KEY], Rmodel("Fdatotal_gentle")],
    })
    l.tasks[-1].n_least_episode = 100

    ############################################################################
    # Execute
    ############################################################################
    if True:
        ExecuteLearning(ct, l)
