from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import pandas as pd
import yaml

def Help():
  pass

def GetSLData(sl_path):
  with open(sl_path, "r") as yml:
    sl = yaml.load(yml)

  envs = defaultdict(list)
  trues = defaultdict(list)
  for ep in range(len(sl)):
    config = sl[ep]["config"]
    reward = sl[ep]["reward"]
    sequence = sl[ep]["sequence"]

    envs["smsz"].append(config["size_srcmouth"][0][0])
    if config["material2"][0][0] == 0.7: envs["mtr"].append("bounce")
    elif config["material2"][2][0] == 0.25: envs["mtr"].append("ketchup")
    elif config["material2"][0][0] == 1.5: envs["mtr"].append("natto")
    else: envs["mtr"].append("nobounce")
    trues["da_spill2"].append(reward[1][2][0]["da_spill2"])
    trues["da_pour"].append(reward[1][3][0]["da_pour"])
    trues["lp_pour_x"].append(sequence[3]["n2b"]["lp_pour"][0][0])
    trues["lp_pour_z"].append(sequence[3]["n2b"]["lp_pour"][2][0])

  return envs, trues

def Run(ct, *args):
  name_log = args[0]
  root_path = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'
  model_path = root_path + name_log + "/models/"
  sl_path = root_path + name_log + "/sequence_list.yaml"

  # envs, trues = GetSLData()
  
  domain= TGraphDynDomain()
  SP= TCompSpaceDef
  domain.SpaceDefs={
    'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position (x,z)
    'dtheta2': SP('action',1,min=[0.002],max=[0.005]),  #Pouring skill parameter for 'std_pour'
    'shake_axis2': SP('action',2,min=[0.01,-0.5*math.pi],max=[0.1,0.5*math.pi]),  #Pouring skill parameter for 'shake_A'
    'ps_rcv': SP('state',12),  #4 edge point positions (x,y,z)*4 of receiver
    'lp_pour': SP('state',3),  #Pouring axis position (x,y,z) in receiver frame
    'da_pour': SP('state',1),  #Amount poured in receiver (displacement)
    'da_spill2': SP('state',1),  #Amount spilled out (displacement)
    'size_srcmouth': SP('state',1),  #Size of mouth of the source container
    }
  domain.Models={
    # 'Fmvtopour2': [  #Move to pouring point
    #   ['p_pour_trg'],
    #   ['lp_pour'],None],
    'Fmvtopour2': [  #Move to pouring point
      ['p_pour_trg'],
      ['da_pour','da_spill2'],None],
    'Fflowc_tip10': [  #Flow control with tipping.
      ['lp_pour','size_srcmouth'],
      ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    'Fflowc_shakeA10': [  #Flow control with shake_A.
      ['lp_pour','size_srcmouth','shake_axis2'],
      ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    }
  
  mm= TModelManager(domain.SpaceDefs, domain.Models)
  mm.Load(LoadYAML(model_path+'model_mngr.yaml'), model_path)
  mm.Init()
  Fmvtopour2 = mm.Models["Fmvtopour2"][2]
  Fflowc_shakeA10 = mm.Models["Fflowc_shakeA10"][2]
  Fflowc_tip10 = mm.Models["Fflowc_tip10"][2]

  # x1 = Fmvtopour2.DataX[:,0]
  # x2 = Fmvtopour2.DataX[:,1]
  

  colors = []
  zorders = []
  markers = []
  miss_and_spills = []
  DataX = Fflowc_tip10.DataX
  DataY = Fflowc_tip10.DataY
  # DataX = Fflowc_shakeA10.DataX
  # DataY = Fflowc_shakeA10.DataY
  x1 = DataX[:,0]
  x2 = DataX[:,2]

  for i,(X,Y) in enumerate(zip(DataX, DataY)):
    # if len(X)!=4:
    #   markers.append("")
    #   colors.append([1,0,0,1])
    #   zorders.append(0)
    #   continue

    v = Y[0]
    # d = abs(Fflowc_tip10.Predict(X).Y[0] - Y[0])
    markers.append("o")
    # if d<0.05: 
    #   # colors.append([0,0,1,0.3])
    #   # zorders.append(-1)
    #   markers.append("o")
    # else: 
    #   # colors.append([1,0,0,1])
    #   # zorders.append(1)
    #   markers.append("*")
    if v>=0.3: 
      colors.append([0,0,1,0.3])
      zorders.append(-1)
    else: 
      colors.append([1,0,0,1])
      zorders.append(1)

    # v = round(Y[1],2)
    # # d = abs(Fflowc_tip10.Predict(X).Y[1] - Y[1])
    # # if d>=0.05:
    # #   markers.append("*")
    # # else:
    # #   markers.append("o")
    # markers.append("o")
    # if v<0.5: 
    #   colors.append([0,0,1,0.3])
    #   zorders.append(-1)
    # # elif v==0.1:
    # #   colors.append([1,0.5,0.25,1])
    # #   zorders.append(1)
    # else: 
    #   colors.append([1,0,0,1])
    #   zorders.append(1)

    # if Y[1]>=0.1: markers.append("*")
    # else        : markers.append("o")
    # if diff>=0.05 and Y[1]>=0.1 : miss_and_spills.append(True)
    # else                        : miss_and_spills.append(False)

  fig = plt.figure(figsize=(20,4))
  fig.suptitle("observed da_pour with sampled point (std_pour)")
  ep_block = 80
  for i in range(int(len(x1)/ep_block)):
    for j in np.linspace(ep_block*i, ep_block*(i+1)-1, ep_block):
      j = int(j)
      fig.add_subplot(1, len(x1)/ep_block, i+1).scatter(x1[j],x2[j],s=30,
                        c=colors[j],
                        zorder=zorders[j],
                        marker=markers[j],
                      )
    plt.xlim(-0.3,0.45)
    plt.ylim(0.1,0.5)
    # plt.title("episode " + str(i*ep_block) + "~" + str((i+1)*ep_block) )
    plt.title("episode " + str(i*ep_block) + "~" + str((i+1)*ep_block) +"\n"
              # + "spilled case: "+str(len(filter(lambda x: x=="*", markers[ep_block*i:ep_block*(i+1)])))+"/"+str(ep_block) +"\n"
              # + "miss predict (diff>0.05) case: "+str(len(filter(lambda x: x[0]==1, colors[ep_block*i:ep_block*(i+1)])))+"/"+str(ep_block) +"\n"
              # + "miss predict and spilled case: "+str(sum(miss_and_spills[ep_block*i:ep_block*(i+1)]))+"/"+str(ep_block)
              )
    plt.xlabel("lp_pour_x")
    plt.ylabel("lp_pour_z")
    # plt.plot(np.linspace(min(min(est), min(true)), max(max(est), max(true)), 2), np.linspace(min(min(est), min(true)), max(max(est), max(true)), 2), c="orange", linestyle="dashed")
  plt.subplots_adjust(left=0.05, right=0.95, top=0.8)
  plt.show()

  # plt.close()
  # fig = plt.figure()
  # plt.title("da_pour histgram")
  # plt.hist(Fflowc_tip10.DataY[:,0], bins=5)
  # plt.show()

  # plt.close()
  # fig = plt.figure(figsize=(20,4))
  # plt.title("minimal smsz change around spilled sample"+"\n"+"smsz = 0.053, p_pour_trg = (0.435, 0.428)")
  # print(len(Fflowc_tip10.DataY[:,0]))
  # print(len(np.linspace(0.065,0.0575,100)))
  # plt.scatter(np.linspace(0.065,0.0575,100),Fflowc_tip10.DataY[:,0])
  # plt.xlim(0.05,0.056)
  # plt.ylim(-0.05,0.2)
  # plt.xlabel("smsz")
  # plt.ylabel("da_spill2")
  # plt.show()