from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import pandas as pd

def Help():
  pass

def Run(ct, *args):
  root_path = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'
  model_path = root_path + args[0] + "/models/"
  target_model = "Fmvtopour2"

  domain= TGraphDynDomain()
  SP= TCompSpaceDef
  domain.SpaceDefs={
    'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position (x,z)
    'shake_axis2': SP('action',2,min=[0.01,-0.5*math.pi],max=[0.1,0.5*math.pi]),  #Pouring skill parameter for 'shake_A'
    'ps_rcv': SP('state',12),  #4 edge point positions (x,y,z)*4 of receiver
    'da_pour': SP('state',1),  #Amount poured in receiver (displacement)
    'da_spill2': SP('state',1),  #Amount spilled out (displacement)
    'size_srcmouth': SP('state',1),  #Size of mouth of the source container
    }
  domain.Models={
    'Fmvtopour2': [  #Move to pouring point
      ['p_pour_trg', "size_srcmouth", "shake_axis2"],
      ['da_pour','da_spill2'],None],
    }
  
  mm= TModelManager(domain.SpaceDefs, domain.Models)
  mm.Load(LoadYAML(model_path+'model_mngr.yaml'), model_path)
  mm.Init()
  model = mm.Models[target_model][2]

  x1 = model.DataX[:,0]
  x2 = model.DataX[:,1]
  # print(len(x1),len(x2))

  # colors = []
  # for y in model.DataY:
  #   y = y[1]
  #   if    y<0.1: colors.append("blue")
  #   elif  y<0.2: colors.append("orange")
  #   else       : colors.append("red")

  # fig = plt.figure()
  # plt.scatter(x1,x2,s=10,c=colors)
  # plt.xlim(0.38,0.45)
  # plt.ylim(0.18,0.28)
  # plt.show()


  colors = []
  markers = []
  miss_and_spills = []
  for i,(X,Y) in enumerate(zip(model.DataX, model.DataY)):
    pred = model.Predict(X).Y.ravel()
    diff = abs(pred - Y)[1]
    # print(i, pred, Y, diff)
    if diff>=0.05: colors.append([1,0,0,1.0])
    else        : colors.append([0,0,1,0.3])
    if Y[1]>=0.1: markers.append("*")
    else        : markers.append("o")
    if diff>=0.05 and Y[1]>=0.1 : miss_and_spills.append(True)
    else                        : miss_and_spills.append(False)
  
  # ep_block = 20
  # for i in range(len(x1)/ep_block):
  #   fig = plt.figure()
  #   plt.scatter(x1[ep_block*i:ep_block*(i+1)],x2[ep_block*i:ep_block*(i+1)],s=10,c=colors[ep_block*i:ep_block*(i+1)])
  #   # plt.xlim(0.38,0.45)
  #   # plt.ylim(0.18,0.28)
  #   plt.xlim(0.3,0.7)
  #   plt.ylim(0.05,0.30)
  #   plt.show()

  fig = plt.figure(figsize=(20,5))
  fig.suptitle("da_spill2 prediction with sampled point")
  ep_block = 50
  for i in range(int(len(x1)/ep_block)):
    for j in np.linspace(ep_block*i, ep_block*(i+1)-1, ep_block):
      j = int(j)
      fig.add_subplot(1, len(x1)/ep_block, i+1).scatter(x1[j],x2[j],s=30,c=colors[j], marker=markers[j])
    plt.xlim(0.3,0.7)
    plt.ylim(0.05,0.30)
    # plt.title("episode " + str(i*ep_block) + "~" + str((i+1)*ep_block) )
    plt.title("episode " + str(i*ep_block) + "~" + str((i+1)*ep_block) +"\n"
              + "spilled case: "+str(len(filter(lambda x: x=="*", markers[ep_block*i:ep_block*(i+1)])))+"/"+str(ep_block) +"\n"
              + "miss predict (diff>0.05) case: "+str(len(filter(lambda x: x[0]==1, colors[ep_block*i:ep_block*(i+1)])))+"/"+str(ep_block) +"\n"
              + "miss predict and spilled case: "+str(sum(miss_and_spills[ep_block*i:ep_block*(i+1)]))+"/"+str(ep_block))
    # plt.xlabel("est "+var)
    # plt.ylabel("true "+var)
    # plt.plot(np.linspace(min(min(est), min(true)), max(max(est), max(true)), 2), np.linspace(min(min(est), min(true)), max(max(est), max(true)), 2), c="orange", linestyle="dashed")
    # plt.legend()
  plt.subplots_adjust(left=0.05, right=0.95, top=0.75)
  plt.show()

  plt.close()
  fig = plt.figure()
  plt.title("sampled smsz histgram")
  plt.hist(model.DataX[:,2], bins=5)
  plt.show()