from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
from scipy.stats import zscore
from matplotlib import pyplot as plt
from matplotlib.ticker import *

def Help():
  pass

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
    }
  return domain

def Run(ct, *args):
  root_path = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'
  model_path = root_path + args[0] + "/models/"
  # model_path = root_path + args[0] + "/manual_train/models/"
  save_path = root_path + args[0] + "/manual_train/models/"

  # batchsize = 10
  # n_epochs = 10
  # nn_options = {
  #   "batchsize": batchsize,           #default 10
  #   "num_max_update": n_epochs*(200/batchsize),     #default 5000
  #   'num_check_stop': 200/batchsize*10,       #default 50
  #   "dropout": True, #default True
  #   'loss_stddev_stop': 1e-5,  #default 1e-3
  #   # 'AdaDelta_rho': 0.9,        #default 0.9
  # }
  nn_options = {
    # "gpu": 0, 
    "batchsize": 10,           #default 10
    "num_max_update": 5000,     #default 5000
    'num_check_stop': 50,       #default 50
    'loss_stddev_stop': 1e-3,  #default 1e-3
    'AdaDelta_rho': 0.9,        #default 0.9
    # 'train_log_file': '{base}train/nn_log-{name}{code}.dat', 
    # "train_batch_loss_log_file": '{base}train/nn_batch_loss_log-{name}{code}.dat',
  }

  domain = createDomain()
  mm= TModelManager(domain.SpaceDefs, domain.Models)
  mm.Load(LoadYAML(model_path+'model_mngr.yaml'), model_path)
  mm.Init()
  # predict_model = "Fflowc_tip10"
  # predict_model = "Fflowc_shakeA10"
  predict_model = "Fmvtopour2"
  model = mm.Models[predict_model][2]
  # model.Load(data={"params": {"nn_params":None,"nn_params_err":None}},base_dir=model.load_base_dir)
  # model.Init()

  DataX = model.DataX
  DataY = model.DataY
  CPrint(3,"DataX shape:",DataX.shape)
  CPrint(3,"DataY shape:",DataY.shape)


  # domain = createDomain()
  # mm= TModelManager(domain.SpaceDefs, domain.Models)
  # mm.Load({"options": {"type": "lwr"}})
  # # mm.Load({"options": {"dnn_hidden_units": [200,200,200]}})
  # mm.Init()
  # predict_model = "Fflowc_tip10"
  # # predict_model = "Fflowc_shakeA10"
  # model = mm.Models[predict_model][2]
  # # model.DataX, model.DataY = DataX, DataY #for NN

  # model.Options.update({
  #   'c_min': 0.01,
  #   'c_max': 1e6,
  #   'c_gain': 0.7,
  #   'f_reg': 0.01
  # })
  # for X, Y in zip(DataX, DataY):
  #   model.Update(list(X),list(Y),not_learn=False)
  # print(len(model.DataX))
  # for i, (X, Y) in enumerate(zip(model.DataX, model.DataY)):
  #   pred = model.Predict(X)

  # model.Options.update(nn_options)
  # model.Update(None, None, not_learn=False)

  # SaveYAML(mm.Save(save_path), save_path+'model_mngr.yaml')

  if True:
    for i, (X, Y) in enumerate(zip(model.DataX, model.DataY)):
      pred = model.Predict(x=X, x_var=0.0, with_var=True, with_grad=True)
      Print(i, X[0]-0.6, Y[0], pred.Y[0])

  def validate_model(out_idx, do_print=False):
    diff = []
    trues = []
    preds = []
    for i, (X, Y) in enumerate(zip(model.DataX, model.DataY)):
      pred = model.Predict(x=X, x_var=0.0, with_var=True, with_grad=True)
      idx = out_idx
      smsz = X[3]
      t_da_pour = Y[0]
      t_da_spill2 = Y[1]
      p_da_pour_mean = pred.Y[0].item()
      p_da_spill2_mean = pred.Y[1].item()
      p_da_pour_sdv = np.sqrt(pred.Var[0,0])
      p_da_spill2_sdv = np.sqrt(pred.Var[1,1])
      mean = pred.Y[idx]
      var = pred.Var[idx,idx]
      diff.append(abs(mean-Y[idx]).item())
      trues.append(Y[idx])
      preds.append(pred.Y[idx])

      if t_da_pour<0.3:
        print(i, list(X), t_da_pour, t_da_spill2, p_da_pour_mean, p_da_spill2_mean, p_da_pour_sdv, p_da_spill2_sdv)

    print(sum(diff)/len(diff))

    return trues, preds, diff

  def hist():
    var_list = []
    for i, (X, Y) in enumerate(zip(model.DataX, model.DataY)):
      var = X[3]
      da_pour = Y[0]
      da_spill2 = Y[1]
      if da_pour<0.3 and da_spill2<1.0:
        var_list.append(var)
    print(float(len(var_list))/len(model.DataX))

    fig = plt.figure(figsize=(5,5))
    # title = "failure sample's smsz histgram (bins=20)"+"\n"+"da_pour<0.3, lp_pour_z<0.17, episode>100"+"\n"+"selected model: "+predict_model
    title = "failure sample's smsz histgram (bins=20)"+"\n"+"da_pour<0.3, episode>100"+"\n"+"selected model: "+predict_model
    plt.title(title)
    plt.hist(var_list, bins=20, 
              # range=(0.03,0.08),
              # range=(0.12,0.5) 
              # weights=np.ones(len(var_list))/len(var_list)
            )
    plt.xlabel("lp_pour_z")
    plt.ylabel("count")
    # plt.ylim(0,20)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(1))
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.subplots_adjust(
      # left=0.05, right=0.95, 
      top=0.85, hspace=0.6)
    plt.show()

  def scatter():
    x_list = []
    y_list = []
    colors = []
    preds = []
    for i, (X, Y) in enumerate(zip(model.DataX, model.DataY)):
      # pred = model.Predict(x=X, x_var=0.0, with_var=True, with_grad=True)
      da_pour = Y[0]
      da_spill2 = Y[1]
      smsz = X[3]
      x, y = X[0], X[3]

      if True:
        if not (-0.1<=x<=0 and round(X[2],3)==0.325 and round(smsz,2)==0.07):
          continue
        print(i)
        x_list.append(x)
        y_list.append(y)
        if da_pour < 0.3:
          if 0.30<X[2]<0.34:
            color = [1,0,0,1]
          else:
            color = [1,0.5,0.25,1]
        else:
          if 0.30<X[2]<0.34:
            color = [0,1,0,0.3]
          else:
            color = [0,1,1,0.3]
        # color = "black"
        colors.append(color)
        # print(X, Y)


    fig = plt.figure(figsize=(5,5))
    # plt.title("da_pour scatter (0.032<p_pour_z<0.33)")
    for i in range(len(x_list)):
      plt.scatter(x_list[i], y_list[i], c=colors[i])
    plt.xlim(-0.1,-0.0)
    plt.ylim(0.03,0.08)
    # plt.axis("off")
    plt.tick_params(bottom=False,
               left=False,
               right=False,
               top=False)
    plt.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)
    # plt.show()
    plt.savefig("/home/yashima/Pictures/tmp/img1",transparent=True)

  def plot(out_var, trues, preds):
    fig = plt.figure(figsize=(20,4))
    # xlabels = np.linspace(0,len(model.DataX)-1, len(model.DataX))
    # colors = ["red" if s>=0.1 else "blue" for s in model.DataY[:,1]]
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("true/estimated "+out_var+" "+predict_model)
    ax.plot(trues, label="trues")
    ax.plot(preds, label="preds (after training)")
    if out_var=="da_pour":
      ax.hlines(0.3, 0, len(trues)+1, label="target value")
    ax.set_xlim(0,len(trues))
    ax.set_xticks(np.arange(0, len(trues)+1, 10))
    ax.set_xticks(np.arange(0, len(trues)+1, 1), minor=True)
    ax.grid(which='minor', alpha=0.4, linestyle='dotted') 
    ax.grid(which='major', alpha=0.9, linestyle='dotted') 
    ax.set_xlabel("samples")
    ax.set_ylabel(out_var)
    plt.legend()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85)
    plt.show()
  

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

  # trues, preds, diff = validate_model(0, do_print=True)
  # plot("da_pour", trues, preds)

  # trues, preds, diff = validate_model(1)
  # plot("da_spill2", trues, preds)

  # hist()

  # scatter()