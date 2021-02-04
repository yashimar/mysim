from core_tool import *
import numpy as np
from matplotlib import pyplot as plt

def Help():
  pass

def RwdPredict(modeldir):
  FRwd= TNNRegression()
  prefix= modeldir
  FRwd.Load(LoadYAML(prefix+'.yaml'), prefix)
  FRwd.Init()

  ex_list = np.linspace(0,0.6,1000)
  d0 = []
  d1 = []
  d2 = []
  for ex in ex_list:
    d0.append([ex, FRwd.Predict(x=[ex], x_var=[0.0**2], with_var=True).Y.item()])
    d1.append([ex, FRwd.Predict(x=[ex], x_var=[0.1**2], with_var=True).Y.item()])
    d2.append([ex, FRwd.Predict(x=[ex], x_var=[0.2**2], with_var=True).Y.item()])
  d0 = np.array(d0)
  d1 = np.array(d1)
  d2 = np.array(d2)

  return d0,d1,d2

def WolframAlphaResult():
  #cos(x), Std[x]=2.0
  # d = np.array([
  #   [-3.0, -0.133981],
  #   [-2.5, -0.1084],
  #   [-2.0, -0.056319],
  #   [-1.5, 0.00957],
  #   [-1.0, 0.073122],
  #   [-0.5, 0.11876],
  #   [0, 0.13535],
  #   [0.5, 0.11876],
  #   [1.0, 0.073122],
  #   [1.5, 0.00957],
  #   [2.0, -0.056319],
  #   [2.5, -0.1084],
  #   [3.0, -0.133981]
  # ])
  #cos(x), Std[x]=1.0
  # d = np.array([
  #   [-3.0, -0.60046],
  #   [-2.5, -0.4859],
  #   [-2.0, -0.252],
  #   [-1.5, 0.0429043],
  #   [-1.0, 0.3277],
  #   [-0.5, 0.532281],
  #   [0, 0.606531],
  #   [0.5, 0.532281],
  #   [1.0, 0.3277],
  #   [1.5, 0.0429043],
  #   [2.0, -0.252],
  #   [2.5, -0.4859],
  #   [3.0, -0.60046]
  # ])
  #-100*max(0,0.3-x)**2 -1.0*max(0,x-0.3)**2, Std[x]=0.1
  d1 = np.array([
    [0, -9.9998],
    [0.1, -4.99429],
    [0.2, -1.9254],
    [0.3, -0.505],
    [0.4, -0.0945],
    [0.5, -0.0557],
    [0.6, -0.099998]
  ])
  #-100*max(0,0.3-x)**2 -1.0*max(0,x-0.3)**2, Std[x]=0.2
  d2 = np.array([
    [0, -12.9095],
    [0.1, -7.70165],
    [0.2, -4.16983],
    [0.3, -2.02],
    [0.4, -0.880171],
    [0.5, -0.378346],
    [0.6, -0.220474]
  ])
  return d1,d2

def Run(ct, *args):
  dat_path = "/tmp"+"/"
  modeldir= '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'\
            +'reward_model/p1_model/FRwd3'

  f = np.loadtxt(dat_path+"true.dat")
  expec_nu0 = np.loadtxt(dat_path+"expec0.dat")
  expec_nu1 = np.loadtxt(dat_path+"expec1.dat")
  expec_nu2 = np.loadtxt(dat_path+"expec2.dat")
  expec_wa1, expec_wa2 = WolframAlphaResult()
  expec_pred0, expec_pred1, expec_pred2 = RwdPredict(modeldir)
  
  fig = plt.figure()
  plt.plot(f[:,0], f[:,1], label="f(x)")
  # plt.plot(expec_nu0[:,0], expec_nu0[:,1], label="Std[x]=0.0 (numerical integration)")
  plt.plot(expec_pred0[:,0], expec_pred0[:,1], label="Std[x]=0.0 (NN model)", linestyle="dashed")
  plt.plot(expec_nu1[:,0], expec_nu1[:,1], label="Std[x]=0.1 (num expec)")
  plt.plot(expec_pred1[:,0], expec_pred1[:,1], label="Std[x]=0.1 (NN model)", linestyle="dashed")
  plt.scatter(expec_wa1[:,0], expec_wa1[:,1], c="red", label="Std[x]=0.1 (wa result)")
  plt.plot(expec_nu2[:,0], expec_nu2[:,1], label="Std[x]=0.2 (num expec)")
  plt.plot(expec_pred2[:,0], expec_pred2[:,1], label="Std[x]=0.2 (NN model)", linestyle="dashed")
  plt.scatter(expec_wa2[:,0], expec_wa2[:,1], c="red", marker="*", label="Std[x]=0.2 (wa result)")
  # plt.ylim(-1.0,0.1)
  plt.title("f(x)=-100*max(0,0.3-x)**2 -1.0*max(0,x-0.3)**2")
  plt.xlabel("x")
  plt.ylabel("E[f]")
  plt.grid(linestyle="dashed", zorder=-1)
  plt.legend()
  plt.show()