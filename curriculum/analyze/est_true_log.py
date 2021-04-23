from core_tool import *
from ..vis.util import FormatSequence
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


def Help():
    pass


def Run(ct, *args):
    path = "curriculum/manual_skill_ordering2/ketchup_0055/first"
    envs, trues, ests = FormatSequence(path)

    es = [e.item() for e in ests["selected"]["da_total"]["mean"]]
    print(es)