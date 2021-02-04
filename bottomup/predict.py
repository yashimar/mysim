import enum
from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
import joblib
from scipy.stats import zscore
from matplotlib import pyplot as plt
from matplotlib.ticker import *

def Help():
  pass

def RwdModel():
  modeldir= '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'\
            +'reward_model'+"/"
  FRwd= TNNRegression()
  prefix= modeldir+'p1_model/FRwd3'
  FRwd.Load(LoadYAML(prefix+'.yaml'), prefix)
  FRwd.Init()

  return FRwd

def createDomain():
  domain= TGraphDynDomain()
  SP= TCompSpaceDef
  domain.SpaceDefs={
    'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position (x,z)
    'dtheta2': SP('action',1,min=[0.002],max=[0.005]),  #Pouring skill parameter for 'std_pour'
    'shake_axis2': SP('action',2,min=[0.01,-0.5*math.pi],max=[0.1,0.5*math.pi]),  #Pouring skill parameter for 'shake_A'
    'ps_rcv': SP('state',12),  #4 edge point positions (x,y,z)*4 of receiver
    'lp_pour': SP('state',3),  #Pouring axis position (x,y,z) in receiver frame
    "da_trg": SP("state",1),
    "a_spill2": SP("state",1),
    "a_src": SP("state",1),
    'da_pour': SP('state',1),  #Amount poured in receiver (displacement)
    'da_spill2': SP('state',1),  #Amount spilled out (displacement)
    'size_srcmouth': SP('state',1),  #Size of mouth of the source container
    }
  domain.Models={
    # 'Fmvtopour2': [  #Move to pouring point
    #   ['p_pour_trg'],
    #   ['lp_pour'],None],
    'Fmvtopour2': [  #Move to pouring point
      ['p_pour_trg','size_srcmouth','shake_axis2'],
      ['da_pour','da_spill2'],None],
    # 'Fflowc_tip10': [  #Flow control with tipping.
    #   ['lp_pour','size_srcmouth'],
    #   ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    # 'Fflowc_shakeA10': [  #Flow control with shake_A.
    #   ['lp_pour','size_srcmouth','shake_axis2'],
    #   ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    'Fflowc_tip10': [  #Flow control with tipping.
      ['lp_pour','size_srcmouth',
        "da_trg","a_src","a_spill2"],
      ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    'Fflowc_shakeA10': [  #Flow control with shake_A.
      ['lp_pour','size_srcmouth','shake_axis2',
        "da_trg","a_src","a_spill2"],
      ['da_pour','da_spill2'],None],  #Removed 'p_pour'
    'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
                  TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2)],
    # 'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
    #               TLocalQuad(3,lambda y:-10.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2)],
    # 'Rdamount':  [['da_pour','da_trg','da_spill2',"skill"],[REWARD_KEY],
    #               TLocalQuad(3,lambda y:-10.0*(0.3-y[0])**2)],
    # 'Rdamount':  [['da_pour','da_trg','da_spill2',"skill"],[REWARD_KEY],
    #               TLocalQuad(3,lambda y:-10.0*y[0]**2)],
    # 'Rdamount':  [['da_pour','da_trg','da_spill2',"skill"],[REWARD_KEY],
    #               TLocalQuad(3,lambda y:np.cos(y[0]))],
    # 'Rdamount': [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
    #              TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - (200.0 if y[3]!=0 else 0.0))]
    # 'Rdamount': [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
    #              TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - (10 if y[3]!=0 else 0.0))]
    # 'Rdamount': [['da_pour','da_trg','da_spill2','skill'],[REWARD_KEY],
    #              TLocalQuad(4,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2)]
    # "Rdamount" : [['da_pour'],[REWARD_KEY],RwdModel()]
    }
  return domain

def Run(ct, *args):
  root_path = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'
  name_log = args[0]
  model_path = root_path + name_log + "/models/"
  tree_path = root_path+name_log+"/best_est_trees/"
  # model_path = root_path + args[0] + "/manual_train/models/"

  domain = createDomain()
  mm= TModelManager(domain.SpaceDefs, domain.Models)
  print(model_path+'model_mngr.yaml')
  mm.Load(LoadYAML(model_path+'model_mngr.yaml'), model_path)
  mm.Init()
  Fmvtopour2 = mm.Models["Fmvtopour2"][2]
  Fflowc_tip10 = mm.Models["Fflowc_tip10"][2]
  Fflowc_shakeA10 = mm.Models["Fflowc_shakeA10"][2]
  Rdamount = mm.Models["Rdamount"][2]
  # print(Rdamount.Options)
  # h = 0.01
  # Rdamount.Load(data={"options": {"h": h, "maxd1": 1e10, "maxd2": 1e10, "tune_h": True}})

  if False:
    x = [0.3, 0.3, 0.0, 0]
    # res = TaylorExp2(Rdamount.MF, x, h=0.1, maxd1=1e10, maxd2=1e10)
    res = TaylorExp2(Rdamount.MF, x)
    print("y: ")
    print(res[0])
    print("dy: ")
    print(res[1])
    print("ddy: ")
    print(res[2])

  if False:
    p_pour_trg = Fmvtopour2.DataX[-1]
    lp_pour = Fmvtopour2.Predict(x=p_pour_trg, x_var=0.0, with_var=True)
    print("lp_pour")
    print("Y:")
    print(lp_pour.Y)
    print("Var:")
    print(lp_pour.Var)
    print("")

    out = Fflowc_tip10.Predict(x=np.array([lp_pour.Y[0], lp_pour.Y[1], lp_pour.Y[2], 0.07]), x_var=np.array([lp_pour.Var[0,0],0.0,lp_pour.Var[2,2],0.0]), with_var=True)
    print("out")
    print("Y:")
    print(out.Y)
    print("Var:")
    print(out.Var)
    print("")

    # r = Rdamount.Predict(x=np.array([out.Y[0], 0.3, out.Y[1]]), x_var=np.array([out.Var[0,0], 0.0, 0.0]), with_var=True)
    # r = Rdamount.Predict(x=np.array([out.Y[0], 0.3, out.Y[1]]), x_var=np.array([0.0, 0.0, 0.0]), with_var=True)
    r = Rdamount.Predict(x=np.array([out.Y[0], 0.3, out.Y[1], 0]), x_var=np.array([out.Var[0,0], 0.0, out.Var[1,1], 0.0]), with_var=True)
    # r = Rdamount.Predict(x=np.array([0.38343143463134766, 0.3, 0.7967915534973145, 0.0]), x_var=np.array([0.010406531393527985, 0.0, 0.14846406877040863, 0.0]), with_var=True)
    print("r")
    print("Y:")
    print(r.Y)
    print("Var:")
    print(r.Var)

  if False:
    with open(tree_path+"ep"+str(619)+"_n0.jb", mode="rb") as f:
      tree = joblib.load(f)
    for key in tree.Tree.keys():
      print(key.A)
      print(tree.Tree[key].XS)

  if True:
    ppour_list = np.linspace(0.0,0.55,100)
    x_list = [np.array([ppour, 0.3, 0]) for ppour in ppour_list]
    x_var_list = np.array([0.0, 0.10, 0.20])**2
    color_list = ["skyblue","green","purple"]
    # x_var_list = np.array([0.1])**2
    y_list_meta = []
    y_var_list_meta = []
    max_list = []
    max_var_list = []
    for x_var in x_var_list:
      y_list = []
      y_var_list = []
      for x in x_list:
        pred = Rdamount.Predict(x=x, x_var=[x_var, 0.0, 0.0], with_var=True)
        mean = pred.Y.item()
        var = pred.Var.item()
        y_list.append(mean)
        y_var_list.append(var)
        if round(x[0], 3)==0.3:
          max_list.append(mean)
          max_var_list.append(var)
      y_list_meta.append(y_list)
      y_var_list_meta.append(y_var_list)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("f(x)=-100*max(0,0.3-x)**2 -1.0*max(0,x-0.3)**2")
    for i, (x_var, y_list, y_var_list) in enumerate(zip(x_var_list, y_list_meta, y_var_list_meta)):
      ax.plot(ppour_list, y_list, label="Std[x]="+str(np.sqrt(x_var))+" (Taylor exp)", c=color_list[i])
      # ax.errorbar(ppour_list, y_list, yerr=np.sqrt(y_var_list), zorder=-i)
    # for v in max_list:
    #   ax.axhline(v)
    # ax.axvline(0.3, linestyle="dashed", c="gray")
    ax.set_ylim(-1, 0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("E[f]")
    plt.grid(linestyle="dashed")
    # ax.set_yscale('symlog')
    # ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.legend()
    plt.show()
    # plt.savefig("/home/yashima/Pictures/tmp/tmp4/0"+str(h).split(".")[1]+".png")
    ## plt.savefig("/home/yashima/Pictures/tmp/tmp4/020.png")
    # plt.close()
    print(max_list)
    print(max_var_list)