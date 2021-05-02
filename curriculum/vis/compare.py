from core_tool import *
from util import FormatSequence
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


SCRATCH = "scratch"
CURRICULUM = "curriculum"


def Help():
    pass


def Run(ct, *args):
    path_list = {
        SCRATCH: [
            # "curriculum/scratch/ketchup_0055/first",
            # "curriculum/scratch/ketchup_0055/second",
            # "curriculum/scratch/ketchup_0055/third",
            # "curriculum/scratch/ketchup_0055/fourth",
            # "curriculum/scratch/ketchup_0055/removed_dtheta1/first",
            # "curriculum/scratch/ketchup_0055/removed_dtheta1/second",
            # "curriculum/scratch/ketchup_0055/removed_dtheta1/third",
            # "curriculum/scratch/ketchup_0055/removed_dtheta1/fourth"
            "curriculum/scratch/curriculum_test/t1/first",
            "curriculum/scratch/curriculum_test/t2/first",
            "curriculum/scratch/curriculum_test/t3/first",
            "curriculum/scratch/curriculum_test/t4/first",
            "curriculum/scratch/curriculum_test/t5/first",
            "curriculum/scratch/curriculum_test/t6/first",
            "curriculum/scratch/curriculum_test/t7/first",
        ],
        CURRICULUM: [
            # "curriculum/manual_skill_ordering/ketchup_0055/first",
            # "curriculum/manual_skill_ordering/ketchup_0055/second",
            # "curriculum/manual_skill_ordering/ketchup_0055/third",
            # "curriculum/manual_skill_ordering/ketchup_0055/fourth",
            # "curriculum/manual_skill_ordering2/ketchup_0055/second",
            # "curriculum/manual_skill_ordering2/ketchup_0055/third",
            # "curriculum/manual_skill_ordering2/ketchup_0055/fourth",
            # "curriculum/manual_skill_ordering2/ketchup_0055/fifth",
            "curriculum/manual_init_state_ordering/t1/first",
            "curriculum/manual_init_state_ordering/t2/first",
            "curriculum/manual_init_state_ordering/t3/first",
            "curriculum/manual_init_state_ordering/t4/first",
            "curriculum/manual_init_state_ordering/t5/first",
            "curriculum/manual_init_state_ordering/t6/first",
        ]
    }
    seq = defaultdict(lambda: defaultdict(list))

    for type in [SCRATCH, CURRICULUM]:
        for path in path_list[type]:
            envs, trues, ests = FormatSequence(path)
            seq[type]["envs"].append(envs)
            seq[type]["trues"].append(trues)
            seq[type]["ests"].append(ests)

        returns = [trues[".r"] for trues in seq[type]["trues"]]
        seq[type]["mean_returns"] = np.mean(returns, axis=0)
        seq[type]["std_returns"] = np.std(returns, axis=0)

    fig = plt.figure(figsize=(16,5))
    plt.errorbar(range(0,len(seq[SCRATCH]["mean_returns"])), seq[SCRATCH]["mean_returns"], yerr=seq[SCRATCH]["std_returns"], c="blue", ecolor="lightblue", capsize=5)
    plt.ylim(-30,0.5)
    plt.xlim(0,120)
    plt.title("scratch")
    plt.xlabel("episode")
    plt.ylabel("return (mean +/- 1SD)")
    plt.show()

    fig = plt.figure(figsize=(16,5))
    # plt.errorbar(range(0,len(seq[SCRATCH]["mean_returns"])), seq[SCRATCH]["mean_returns"], yerr=seq[SCRATCH]["std_returns"], c="blue", ecolor="lightblue")
    # ts = [0, 9, 14, 19, 24, 29, 34, len(seq[CURRICULUM]["mean_returns"])]
    # ts = [0, 11, 19, 27, 35, 43, 51, 59, len(seq[CURRICULUM]["mean_returns"])]
    # ts = [0, 11, 19, 27, 35, 43, 51, 59]
    ts = [0, 20, 60, 80, len(seq[CURRICULUM]["mean_returns"])]
    c = "gray"
    alpha = 0.2
    for i in range(1,len(ts)):
        ss = ts[i-1]
        se = ts[i]
        mr = seq[CURRICULUM]["mean_returns"][ss:se]
        ms = seq[CURRICULUM]["std_returns"][ss:se]
        plt.errorbar(np.arange(ss,se)+0.1, mr, yerr=ms, c="red", ecolor="pink", capsize=5)
        plt.vlines(ss, -100, 100, linestyles="dashed", color=c, zorder=-10, alpha=alpha)
    plt.ylim(-30,0.5)
    plt.xlim(0,120)
    plt.title("curriculum")
    plt.xlabel("episode")
    plt.ylabel("return (mean +/- 1SD)")
    plt.show()