from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
from scipy.stats import zscore
from matplotlib import pyplot as plt

def Help():
  pass

def createDomain():
  domain= TGraphDynDomain()
  SP= TCompSpaceDef
  domain.SpaceDefs={
    'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),
    'da_spill2': SP('state',1),
    'da_pour':  SP('state',1),
    }
  domain.Models={
    'Fmvtopour2': [  #Move to pouring point
      ['p_pour_trg'],
      ['da_pour','da_spill2'],None],
    }
  return domain

def Run(ct, *args):
  root_path = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'
  model_path = root_path + args[0] + "/models/"
  # model_path = root_path + args[0] + "/manual_train/models/"
  save_path = root_path + args[0] + "/manual_train/models/"

  batchsize = 10
  n_epochs = 10
  nn_options = {
    "batchsize": batchsize,           #default 10
    "num_max_update": n_epochs*(200/batchsize),     #default 5000
    'num_check_stop': 200/batchsize*10,       #default 50
    "dropout": True, #default True
    'loss_stddev_stop': 1e-5,  #default 1e-3
    # 'AdaDelta_rho': 0.9,        #default 0.9
  }

  domain = createDomain()
  mm= TModelManager(domain.SpaceDefs, domain.Models)
  mm.Load(LoadYAML(model_path+'model_mngr.yaml'), model_path)
  mm.Init()
  model = mm.Models["Fmvtopour2"][2]

  DataX = model.DataX
  DataY = model.DataY
  CPrint(3,"DataX shape:",DataX.shape)
  CPrint(3,"DataY shape:",DataY.shape)


  # domain = createDomain()
  # domain= TGraphDynDomain()
  # SP= TCompSpaceDef
  # domain.SpaceDefs={
  #   'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),
  #   'p_pour_trg2': SP('action',2,min=[0.2**2,0.1**2],max=[1.2**2,0.7**2]),
  #   'r': SP('action',1,min=[0.],max=[1e5]),
  #   'da_spill2': SP('state',1),
  #   'da_pour':  SP('state',1),
  #   }
  # domain.Models={
  #   'Fmvtopour2': [  #Move to pouring point
  #     ['p_pour_trg'],
  #     ['da_spill2'],None],
  #   }
  # mm= TModelManager(domain.SpaceDefs, domain.Models)
  # # mm.Load({"options": {"type": "lwr"}})
  # mm.Load({"options": {"dnn_hidden_units": [200,200]}})
  # mm.Init()
  # model = mm.Models["Fmvtopour2"][2]

  # rs = [x[0]**2+x[1]**2 for x in DataX]
  # DataX = np.insert(DataX, -1, rs, axis=1)
  # DataX = np.log(DataX)
  # DataX, DataY = DataX[50:], DataY[50:]
  # DataX = zscore(DataX)
  # DataY = DataY[:,1].reshape(-1,1)
  # model.DataX, model.DataY = DataX, DataY
  # model.DataX.extend(DataX)
  # model.DataY.extend(DataY)

  # model.Options.update({
  #   'c_min': 0.01,
  #   'c_max': 1e6,
  #   'c_gain': 0.7,
  #   'f_reg': 0.01
  # })
  # for X, Y in zip(DataX, DataY):
  #   model.Update(list(X),list(Y),not_learn=False)

  model.Options.update(nn_options)
  model.Update(None, None, not_learn=False)

  # SaveYAML(mm.Save(save_path), save_path+'model_mngr.yaml')

  diff = []
  for i, (X, Y) in enumerate(zip(model.DataX, model.DataY)):
    pred = model.Predict(x=X, x_var=0.0, with_var=True, with_grad=True)
    mean = pred.Y[1]
    var = pred.Var[1,1]
    diff.append(abs(mean-Y[1]).item())
    # print(i, list(X), Y[1], mean, np.sqrt(var))
    # mean = pred.Y[0]
    # var = pred.Var[0,0]
    # diff.append(abs(mean-Y[0]))
    # print(i, list(X), Y[0], mean, np.sqrt(var))
  print(sum(diff)/len(diff))

  # fig = plt.figure(figsize=(20,5))
  # xlabels = np.linspace(0,len(model.DataX)-1, len(model.DataX))
  # colors = ["red" if s>=0.1 else "blue" for s in model.DataY[:,1]]
  # plt.title("prediction error")
  # plt.bar(xlabels, diff, color=colors)
  # plt.ylim(0,0.3)
  # plt.xlabel("sample id")
  # plt.ylabel("|true - pred|")
  # plt.show()


  # diffs = []
  # for i in range(5):
  #   domain = createDomain()
  #   mm= TModelManager(domain.SpaceDefs, domain.Models)
  #   mm.Load(LoadYAML(model_path+'model_mngr.yaml'), model_path)
  #   mm.Init()
  #   model = mm.Models["Fmvtopour2"][2]

  #   model.Options.update(nn_options)
  #   model.Update(None, None, not_learn=False)

  #   diff = []
  #   for i, (X, Y) in enumerate(zip(model.DataX, model.DataY)):
  #     pred = model.Predict(x=X, x_var=0.0, with_var=True, with_grad=True)
  #     mean = pred.Y[1]
  #     var = pred.Var[1,1]
  #     diff.append(abs(mean-Y[1]).item())

  #   diffs.append(diff)

  # diffs = np.array(diffs).reshape(len(model.DataX),-1)
  # diff_mean = [np.mean(diff) for diff in diffs]
  # diff_std = [np.std(diff) for diff in diffs]

  # fig = plt.figure(figsize=(20,5))
  # xlabels = np.linspace(0,len(model.DataX)-1, len(model.DataX))
  # colors = ["red" if s>=0.1 else "blue" for s in model.DataY[:,1]]
  # plt.title("mean +/- std abs diff with 5 samples" + "\n" + " after learning 50 epochs")
  # plt.bar(xlabels, diff_mean, color=colors, yerr=diff_std)
  # plt.ylim(0,0.3)
  # plt.xlabel("sample id")
  # plt.ylabel("|true - pred|")
  # plt.show()

