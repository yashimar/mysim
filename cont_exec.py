from core_tool import *


def Help():
    pass


def Run(ct, *args):
    pass
    for i in range(1,41):
        print(i)
        ct.Run('mysim.curriculum.analyze.log.curriculum5.c1.trues_sampling.tip_ketchup_smsz_dtheta2.opttest.check', "t0.1/5000/t{}".format(i))
        # ct.Run('mysim.curriculum.analyze.log.curriculum5.c1.trues_sampling.tip_ketchup_smsz_dtheta2.opttest.learn', "t0.1/8000/t{}".format(i))
    