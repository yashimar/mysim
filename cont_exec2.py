from core_tool import *


def Help():
    pass


def Run(ct, *args):
    pass
    for i in range(2,11):
        print(i)
        ct.Run('mysim.curriculum.analyze.log.curriculum5.c1.trues_sampling.tip_ketchup_smsz_dtheta2.opttest.onpolicy', "onpolicy/Er/t{}".format(i))
        