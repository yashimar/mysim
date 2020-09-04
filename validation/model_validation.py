#!/usr/bin/python
from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import pandas as pd

def Help():
  return '''Dynamic Planning/Learning for grasping and pouring in ODE simulation
    using DPL version 4 (DNN, Bifurcation).
    Based on tsim.dplD14, modified for the new action system (cf. tsim2.test_dpl1a).
    The behavior is the same as tsim.dplD14.
    We share the same model Famount_* in different skills.
  Usage: tsim2.dplD20'''

def autolabel(rects,ax,x_values_std):
  for i,rect in enumerate(rects):
    width = rect.get_width()
    ax.annotate('{}'.format(str(round(width,5))+" +/- "+str(round(x_values_std[i],5))),
                  xy=(width+0.2,rect.get_y()),
                  xytext=(0,3),
                  textcoords="offset points",
                  ha='center', va='bottom',fontsize=6)

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

def Execute(l):
  node_io = {
    "Fgrasp": [0,1], 
    "Fmvtorcv": [1,4], 
    "Fmvtorcv_rcvmv": [1,2], 
    "Fmvtopour2": [4,5], 
    "Fflowc_tip10": [7,8],
    "Fflowc_shakeA10": [7,8],
    "Famount4": [8,9]
  }
  database = LoadYAML(l.database_path)['Entry']

  # episode_list = [0]
  # dynamics_list = ['Fmvtorcv']
  dynamics_list = l.dynamics_list
  episode_list = l.episode_list
  
  fig1 = plt.figure(figsize=(10,11))
  fig1.suptitle("/".join(l.database_path.split("/")[-4:-1]) +"\n" \
                + "pred_y - true_y MAE of {}episodes each dynamics mean model".format(len(episode_list)), 
                fontsize=10)
  
  fig2 = plt.figure(figsize=(10,11))
  fig2.suptitle("/".join(l.database_path.split("/")[-4:-1]) +"\n" \
                + "skill params l2 norm average of {}episodes each dynamics mean model".format(len(episode_list)), 
                fontsize=10)  
  
  episode_pred_list = defaultdict()
  for episode in episode_list:
    episode_pred_list[episode] = defaultdict()
  out_dims = defaultdict()
  for dynamics in dynamics_list:
    out_dims[dynamics] = defaultdict()

  for di,dynamics in enumerate(dynamics_list):
    CPrint(2,"======== ",dynamics,"========")
    var_i = l.dpl.MM.Models[dynamics][0]
    var_o = l.dpl.MM.Models[dynamics][1]
    Print("Input:",var_i)
    Print("Output:",var_o)
    node_i = node_io[dynamics][0]
    node_o = node_io[dynamics][1]
    diff_list = []
    diff_var_list = defaultdict()
    var_list = defaultdict()
    for var in var_o:
      diff_var_list[var] = []
      var_list[var] = {"key":[],"true":[],"pred":[]}

    for idx,episode in enumerate(episode_list):
      CPrint(2,"----- episode:",str(episode),"-----")
      data = database[episode]["Seq"]
      if idx==0: 
        Print(data[node_i]["Name"],"->",data[node_o]["Name"])

      xs_i = data[node_i]["XS"]
      xs_o = data[node_o]["XS"]
      x = [x for i in range(len(var_i)) for x in xs_i[var_i[i]]["X"]]
      y = [y for i in range(len(var_o)) for y in xs_o[var_o[i]]["X"]]
      
      model = l.dpl.MM.Models[dynamics][2]
      pred = model.Predict(x,x_var=None,with_var=True,with_grad=True)
      episode_pred_list[episode][dynamics] = pred.Y.ravel()
      CPrint(3,"Pred mean:",pred.Y.ravel())
      # pred2 = model.Predict(x,with_var=True)
      # CPrint(3,"Pred var:",pred.Var)
      # CPrint(3,pred.Y.ravel()==pred2.Y.ravel())
      CPrint(3,"True:",sum(y,[]))
      if episode==0 and dynamics=="Famount4":
        print(x)
        # CPrint(3,"real estimate rdamount:",data[-1]["XS"][".r"]["X"][0][0])
        # CPrint(3,"calc estimate rdamount:",-100*max(0.3-pred.Y.ravel()[0],0)**2-max(pred.Y.ravel()[0]-0.3,0)**2-max(pred.Y.ravel()[1],0)**2)
      diff = np.linalg.norm(sum(y,[])-pred.Y.ravel(), ord=2)
      # CPrint(3,"diff:",diff)
      diff_list.append(diff)
      old_dim = 0
      dims = []
      for var_count in range(len(var_o)):
        var = var_o[var_count]
        dim = l.dpl.d.SpaceDefs[var].D
        y_var_component = sum(y,[])[old_dim:old_dim+dim]
        pred_var_component = pred.Y.ravel()[old_dim:old_dim+dim]
        diff_var = np.linalg.norm(y_var_component-pred_var_component, ord=2)
        diff_var_list[var].append(diff_var)
        var_list[var]["true"].append(np.linalg.norm(y_var_component,ord=2))
        var_list[var]["pred"].append(np.linalg.norm(pred_var_component,ord=2))
        old_dim = dim
        dims.append(dim)
        out_dims[dynamics][var] = dim

    x_values ={"keys":[],"values":[np.mean(diff_list)]} 
    # x_values = [np.mean(diff_list)]
    x_values_std = {"keys":[],"values":[np.std(diff_list)]}
    # x_values_std = [np.std(diff_list)]
    x_values2 = {"keys":[],"true":[],"pred":[]}
    x_values2_std = {"keys":[],"true":[],"pred":[]}
    for var in var_o:
      x_values["keys"].append(var)
      x_values["values"].append(np.mean(diff_var_list[var]))
      x_values_std["keys"].append(var)
      x_values_std["values"].append(np.std(diff_var_list[var]))
      x_values2["keys"].append(var)
      x_values2["true"].append(np.mean(var_list[var]["true"]))
      x_values2["pred"].append(np.mean(var_list[var]["pred"]))
      x_values2_std["keys"].append(var)
      x_values2_std["true"].append(np.std(var_list[var]["true"]))
      x_values2_std["pred"].append(np.std(var_list[var]["pred"]))
      
    # try: print(var_list["da_total"]["true"])
    # except: pass
    ax = fig1.add_subplot(len(dynamics_list),1,1+di)
    rect = ax.barh(
        ["all variables"]+x_values["keys"],
        x_values["values"],
        xerr=x_values_std["values"], 
        color=["orange"]+["pink"]*len(diff_var_list)
        )
    ax_title=" | params(dims): "
    for j in range(len(var_o)):
      ax_title += var_o[j]+"({}), ".format(dims[j])
    ax_title = dynamics+" "+ax_title
    ax.set_title(ax_title,fontsize=8)
    ax.set_xlim(0,3.0)
    ax.set_xlabel("MAE of {}episodes".format(len(episode_list)),
                  fontsize=7)
    ax.tick_params("y",labelsize=7)
    ax.tick_params("x",labelsize=6)
    fig1.subplots_adjust(hspace=0.6)
    autolabel(rect,ax,x_values_std["values"])

    ax = fig2.add_subplot(len(dynamics_list),1,1+di)
    x_ticks = [com for t,p in zip(x_values2["true"],x_values2["pred"]) for com in [t,p]]
    x_ticks_err = [com for t,p in zip(x_values2_std["true"],x_values2_std["pred"]) for com in [t,p]]
    y_ticks = [com+"_"+case for com in x_values2["keys"] for case in ["true","pred"]]
    # print(var_list)
    # print(var_list.keys())
    rect = ax.barh(
        y_ticks,
        x_ticks,
        xerr=x_ticks_err, 
        color=["red","pink"]*len(var_list)
        )
    ax.set_title(ax_title,fontsize=8)
    ax.set_xlim(0,3.0)
    ax.set_xlabel("skill params average of {}episodes".format(len(episode_list)),
                  fontsize=7)
    ax.tick_params("y",labelsize=7)
    ax.tick_params("x",labelsize=6)
    fig2.subplots_adjust(hspace=0.6)
    autolabel(rect,ax,x_ticks_err)

  fig1.subplots_adjust(left=0.105,right=0.92,bottom=0.05,top=0.90+0.02)
  # fig1.show()
  if not l.show:plt.close()
  if l.save: fig1.savefig("/home/yashima/Pictures/mtr_sms/model_validation/mean_model_loss/"+l.target_skill+"_"+l.target_mtr_sms+"_"+l.target_type+".png")

  fig2.subplots_adjust(left=0.105,right=0.92,bottom=0.05,top=0.90+0.02)
  # fig2.show()
  if not l.show:plt.close()
  if l.save: fig2.savefig("/home/yashima/Pictures/mtr_sms/model_validation/skill_params/"+l.target_skill+"_"+l.target_mtr_sms+"_"+l.target_type+".png")  


  episode = 5
  data = database[episode]["Seq"][0]["XS"]
  for dynamics in dynamics_list:
    Print("----",dynamics,"----")
    In,Out,model = l.dpl.MM.Models[dynamics]
    Print("In:",In)
    for key in In:
      Print(" ",key,":",data[key]["X"])
    x_in = [data[key]["X"][i] for key in In for i in range(len(data[key]["X"]))]
    dims = DimsXSSA(l.dpl.d.SpaceDefs,In)
    D = sum(dims)
    cov_e= np.zeros((D,D))
    i= 0
    for key,dim in zip(In,dims):
      i2= i+dim
      cov_k, cov_k_is_zero= RegularizeCov(data[key]["Cov"], dim)
      if not cov_k_is_zero:  cov_e[i:i2,i:i2]= cov_k
      i= i2
    pred = model.Predict(x_in,x_var=cov_e,with_var=True,with_grad=True)
    CPrint(3,"Pred:",pred.Y.ravel())
    Print("Out:",Out)
    dim_old = 0
    for key in Out:
      dim = out_dims[dynamics][key]
      data[key] = {"X": pred.Y[dim_old:dim_old+dim], "Cov":np.diag(np.diag(pred.Var)[dim_old:dim_old+dim])}
      dim_old += dim
      Print(" ",key,":",data[key]["X"])  
  CPrint(2,"rdamount:",-100*max(0.3-pred.Y.ravel()[0],0)**2-max(pred.Y.ravel()[0]-0.3,0)**2-max(pred.Y.ravel()[1],0)**2)
  

def Run(ct,*args):
  l= TContainer(debug=True)
  l.show = False
  l.save = False
  l.target_type = args[0]
  l.target_skill = args[1]
  l.target_mtr_sms = args[2]
  target_dir = "mtr_sms/infer/"+l.target_type+"/"+l.target_skill+"/"+l.target_mtr_sms
  base_modeldir = "mtr_sms/learn/"+l.target_type
  l.episode_list = np.arange(0,10)
  dynamics_list = ['Fgrasp','Fmvtorcv','Fmvtorcv_rcvmv','Fmvtopour2',
                    'Fflowc_tip10',
                    # 'Fflowc_shakeA10',
                    'Famount4']

  root_target_dir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'
  root_modeldir = '/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/'

  opt_conf={
    'interactive': False,
    'not_learn': False,
    "model_dir": root_modeldir + base_modeldir + "/models/", 
    'model_dir_persistent': False,
    "db_src": "", 
    'config': {},  #Config of the simulator
    'dpl_options': {
      'opt_log_name': None,  #Not save optimization log.
      },
    }
  
  l.opt_conf= opt_conf
  l.logdir = ""
  l.database_path = root_target_dir+target_dir+'/database.yaml'
  l.dynamics_list = dynamics_list
  l.interactive= l.opt_conf['interactive']
  l.not_learn= l.opt_conf['not_learn']
  #l.not_learn= True  #Models are not trained.

  #Setup dynamic planner/learner
  domain= TGraphDynDomain()
  SP= TCompSpaceDef
  domain.SpaceDefs={
    'skill': SP('select',num=2),  #Skill selection
    'ps_rcv': SP('state',12),  #4 edge point positions (x,y,z)*4 of receiver
    'gh_ratio': SP('action',1,min=[0.0],max=[1.0]),  #Gripper height (ratio)
    'gh_abs': SP('state',1),  #Gripper height (absolute value)
    'p_pour_trg0': SP('state',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position of preparation before pouring (x,z)
      #NOTE: we stopped to plan p_pour_trg0
    'p_pour_trg': SP('action',2,min=[0.2,0.1],max=[1.2,0.7]),  #Target pouring axis position (x,z)
    'dtheta1': SP('action',1,min=[0.01],max=[0.02]),  #Pouring skill parameter for all skills
    'dtheta2': SP('action',1,min=[0.002],max=[0.005]),  #Pouring skill parameter for 'std_pour'
    #'dtheta1': SP('state',1),  #Pouring skill parameter for all skills
    #'dtheta2': SP('state',1),  #Pouring skill parameter for 'std_pour'
    'shake_spd': SP('action',1,min=[0.7],max=[0.9]),  #Pouring skill parameter for 'shake_A'
    #'shake_spd': SP('state',1),  #Pouring skill parameter for 'shake_A'
    #'shake_axis': SP('action',2,min=[0.0,0.0],max=[0.1,0.1]),  #Pouring skill parameter for 'shake_A'
    'shake_axis2': SP('action',2,min=[0.05,-0.5*math.pi],max=[0.1,0.5*math.pi]),  #Pouring skill parameter for 'shake_A'
    #'shake_axis2': SP('state',2),  #Pouring skill parameter for 'shake_A'
    'shake_spd_B': SP('action',1,min=[2.0],max=[8.0]),  #Pouring skill parameter for 'shake_B'
    "shake_range" : SP('action',1,min=[0.02],max=[0.06]),  #Pouring skill parameter for 'shake_B'
    'p_pour': SP('state',3),  #Pouring axis position (x,y,z)
    'lp_pour': SP('state',3),  #Pouring axis position (x,y,z) in receiver frame
    'dps_rcv': SP('state',12),  #Displacement of ps_rcv from previous time
    'v_rcv': SP('state',1),  #Velocity norm of receiver
    #'p_flow': SP('state',2),  #Flow position (x,y)
    'lp_flow': SP('state',2),  #Flow position (x,y) in receiver frame
    'flow_var': SP('state',1),  #Variance of flow
    'a_pour': SP('state',1),  #Amount poured in receiver
    'a_spill2': SP('state',1),  #Amount spilled out
    'a_total':  SP('state',1),  #Total amount moved from source
    'a_trg': SP('state',1),  #Target amount
    'da_pour': SP('state',1),  #Amount poured in receiver (displacement)
    'da_spill2': SP('state',1),  #Amount spilled out (displacement)
    'da_total':  SP('state',1),  #Total amount moved from source (displacement)
    'da_trg': SP('state',1),  #Target amount (displacement)
    'size_srcmouth': SP('state',1),  #Size of mouth of the source container
    'material2': SP('state',4),  #Material property (e.g. viscosity)
    REWARD_KEY:  SP('state',1),
    }
  domain.Models={
    #key:[In,Out,F],
    'Fnone': [[],[], None],
    'Fgrasp': [['gh_ratio'],['gh_abs'],None],  #Grasping. NOTE: removed ps_rcv
    'Fmvtorcv': [  #Move to receiver
      ['ps_rcv','gh_abs','p_pour','p_pour_trg0'],
      ['ps_rcv','p_pour'],None],
    'Fmvtorcv_rcvmv': [  #Move to receiver: receiver movement
      ['ps_rcv','gh_abs','p_pour','p_pour_trg0'],
      ['dps_rcv','v_rcv'],None],
    'Fmvtopour2': [  #Move to pouring point
      ['ps_rcv','gh_abs','p_pour','p_pour_trg'],
      ['lp_pour'],None],
    'Fflowc_tip10': [  #Flow control with tipping.
      ['gh_abs','lp_pour',  #Removed 'p_pour_trg0','p_pour_trg'
       'da_trg','size_srcmouth','material2',
       'dtheta1','dtheta2'],
      ['da_total','lp_flow','flow_var'],None],  #Removed 'p_pour'
    'Fflowc_shakeA10': [  #Flow control with shake_A.
      ['gh_abs','lp_pour',  #Removed 'p_pour_trg0','p_pour_trg'
       'da_trg','size_srcmouth','material2',
       'dtheta1','shake_spd','shake_axis2'],
      ['da_total','lp_flow','flow_var'],None],  #Removed 'p_pour'
    'Fflowc_shakeB10': [  #Flow control with shake_B.
      ['gh_abs','lp_pour',  #Removed 'p_pour_trg0','p_pour_trg'
       'da_trg','size_srcmouth','material2',
       'dtheta1','shake_spd_B','shake_range'],
      ['da_total','lp_flow','flow_var'],None],  #Removed 'p_pour'
    'Famount4': [  #Amount model common for tip and shake.
      ['lp_pour',  #Removed 'gh_abs','p_pour_trg0','p_pour_trg'
       'da_trg','material2',  #Removed 'size_srcmouth'
       'da_total','lp_flow','flow_var'],
      ['da_pour','da_spill2'],None],
    'Rrcvmv':  [['dps_rcv','v_rcv'],[REWARD_KEY],TLocalQuad(13,lambda y:-(np.dot(y[:12],y[:12]) + y[12]*y[12]))],
    'Rmvtopour':  [['p_pour_trg','p_pour'],[REWARD_KEY],TLocalQuad(5,lambda y:-0.1*((y[0]-y[2])**2+(y[1]-y[4])**2))],
    'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
                  TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 1.0*max(0.0,y[2])**2)],
    # 'Rdamount':  [['da_pour','da_trg','da_spill2'],[REWARD_KEY],
    #               TLocalQuad(3,lambda y:-100.0*max(0.0,y[1]-y[0])**2 - 1.0*max(0.0,y[0]-y[1])**2 - 10.0*max(0.0,y[2])**2)],
    'P1': [[],[PROB_KEY], TLocalLinear(0,1,lambda x:[1.0],lambda x:[0.0])],
    'P2':  [[],[PROB_KEY], TLocalLinear(0,2,lambda x:[1.0]*2,lambda x:[0.0]*2)],
    'Pskill': [['skill'],[PROB_KEY], TLocalLinear(0,2,lambda s:Delta1(2,s[0]),lambda s:[0.0]*2)],
    }
  domain.Graph={
      'n0': TDynNode(None,'P1',('Fgrasp','n1')),
      'n1': TDynNode('n0','P2',('Fmvtorcv','n2a'),('Fmvtorcv_rcvmv','n1rcvmv')),
      'n1rcvmv': TDynNode('n1','P1',('Rrcvmv','n1rcvmvr')),
      'n1rcvmvr': TDynNode('n1rcvmv'),
      'n2a': TDynNode('n1','P1',('Fmvtopour2','n2b')),
      'n2b': TDynNode('n2a','P2',('Fnone','n2c'),('Rmvtopour','n2br')),
      'n2br': TDynNode('n2b'),
      'n2c': TDynNode('n2b','Pskill',('Fflowc_tip10','n3ti'),('Fflowc_shakeA10','n3sa')),
      #Tipping:
      'n3ti': TDynNode('n2c','P1',('Famount4','n4ti')),
      'n4ti': TDynNode('n3ti','P1',('Rdamount','n4tir')),
      'n4tir': TDynNode('n4ti'),
      #Shaking-A:
      'n3sa': TDynNode('n2c','P1',('Famount4','n4sa')),
      'n4sa': TDynNode('n3sa','P1',('Rdamount','n4sar')),
      'n4sar': TDynNode('n4sa'),
    }

  # def LogDPL(l):
  #   SaveYAML(l.dpl.MM.Save(l.dpl.MM.Options['base_dir']), l.dpl.MM.Options['base_dir']+'model_mngr.yaml')
  #   SaveYAML(l.dpl.DB.Save(), l.logdir+'database.yaml')
  #   SaveYAML(l.dpl.Save(), l.logdir+'dpl.yaml')

  # if l.interactive and 'log_dpl' in ct.__dict__ and (CPrint(1,'Restart from existing DPL?'), AskYesNo())[1]:

  mm_options= {
      #'type': 'lwr',
      'base_dir': l.logdir+'models/',
    }
  mm= TModelManager(domain.SpaceDefs, domain.Models)
  mm.Load({'options':mm_options})
  if l.opt_conf['model_dir'] not in ('',None):
    if os.path.exists(l.opt_conf['model_dir']+'model_mngr.yaml'):
      mm.Load(LoadYAML(l.opt_conf['model_dir']+'model_mngr.yaml'), l.opt_conf['model_dir'])
    if l.opt_conf['model_dir_persistent']:
      mm.Options['base_dir']= l.opt_conf['model_dir']
    else:
      mm.Options['base_dir']= mm_options['base_dir']
  db= TGraphEpisodeDB()
  if l.opt_conf['db_src'] not in ('',None):
    db.Load(LoadYAML(l.opt_conf['db_src']))

  l.dpl= TGraphDynPlanLearn(domain, db, mm)
  l.restarting= False

  dpl_options={
    'base_dir': l.logdir,
    }
  InsertDict(dpl_options, l.opt_conf['dpl_options'])
  l.dpl.Load({'options':dpl_options})


  if not l.restarting:
    l.dpl.MM.Init()
    l.dpl.Init()

  ct.log_dpl= l.dpl  #for log purpose

  # print 'Copying',PycToPy(__file__),'to',PycToPy(l.logdir+os.path.basename(__file__))
  # CopyFile(PycToPy(__file__),PycToPy(l.logdir+os.path.basename(__file__)))

  Execute(l)
  # LogDPL(l)

  l= None
  return True
