from core_tool import *
from ay_py.core import *
import sys
sys.path.append("/home/yashima/ros_ws/ay_tools/ay_test/python/models/")
import numpy as np
from num_expec import *
from matplotlib import pyplot as plt


def Help():
  pass

def Run(ct,*args):
      
  modeldir= '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'\
            +'reward_model'+"/"

  def Reward(x_in):
    #In=[E[x], Var[x]]
    # ex, varx = x_in
    # R = -100*max(0,0.3-x)**2 -1.0*max(0,x-0.3)**2

    # ex = np.mat([ex]).T
    # varx = np.mat([varx])
    # R_expec = NumExpec(f=R, x0=ex, var=varx, nsd=np.linspace(0,3,1000))
    return R

  def LearnReward():
    #I/O: [E[x], Var[x]],[REWARD_KEY]
    dim_in = 2
    dim_out = 1
    options={
      'base_dir': modeldir+'p1_model/Fflowedout'+"/",
      'n_units': [dim_in] + [128] + [dim_out],
      'name': 'FRwd',
      'loss_stddev_stop': 1.0e-6,
      'loss_stddev_init': 2000.0,
      'num_max_update': 10000,
      "batchsize": 64,
      'num_check_stop': 1000,
      }
    prefix= modeldir+'p1_model/Fflowedout'
    FRwd= TNNRegression()
    FRwd.Load(data={'options':options})
    if os.path.exists(prefix+'.yaml'):
      FRwd.Load(LoadYAML(prefix+'.yaml'), prefix)
      FRwd.Load(data={'options':options}, base_dir=prefix)
    FRwd.Init()
    Print("len(DataX): ", len(FRwd.DataX))

    for i in range(1):
      # ex = [Rand(0.,0.55), Rand(-0.5,1.0)] #Should not contain sdv_x.
      tmp = Rand(0.25,0.35)
      ex = [tmp, tmp+Rand(-0.2,0.2)]
      R = [-100*max(0,ex[0]-ex[1])**2 -1.0*max(0,ex[1]-ex[0])**2]
      FRwd.Update(ex, R, not_learn=True)
    FRwd.UpdateBatch()

    SaveYAML(FRwd.Save(prefix+"/"), prefix+'.yaml')

  def Predict():
    FRwd= TNNRegression()
    prefix= modeldir+'p1_model/Fflowedout'
    FRwd.Load(LoadYAML(prefix+'.yaml'), prefix)
    FRwd.Init()

    plt.close()
    fig = plt.figure()
    # xx = [np.random.normal(x, 0.3, 1) for x in FRwd.DataX]
    # xx = [x for x in FRwd.DataX]
    # plt.scatter(FRwd.DataX, [-100*max(0,0.3-x)**2 -1.0*max(0,x-0.3)**2 for x in xx])
    ex_list = np.array([[0.3, x] for x in np.linspace(0.0,0.55,1000)])
    sdvx_list = [[sdv**2, 0.] for sdv in [0.001, 0.05, 0.1, 0.2]]
    for sdvx in sdvx_list:
      ey_list = []
      vary_list = []
      for ex in ex_list:
        pred = FRwd.Predict(x=ex, x_var=sdvx, with_var=True)
        ey = pred.Y.item()
        vary = pred.Var.item()
        ey_list.append(ey)
        vary_list.append(vary)
      plt.plot(ex_list[:,1], ey_list, label="Sdv[x]="+str(np.sqrt(sdvx[0])))
    # plt.ylim(-1,0.3)
    plt.legend()
    plt.xlabel("E[x]")
    plt.ylabel("Pred E[R(E[x])]")
    plt.show()


  LearnReward()
  Predict()