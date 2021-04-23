#!/usr/bin/python
from core_tool import *
SmartImportReload('tsim.dpl_cmn')
from tsim.dpl_cmn import *
from matplotlib import pyplot as plt
def Help():
  return '''Test of using actions for ODE grasping and pouring simulation (ver.2.1).
    Replaying an episode to check the simulation consistency.
    Based on tsim2.test_act
  Usage: tsim2.test_replay'''


def TestConfigCallback(ct,l,sim):
  for key,value in l.opt_conf['config'].iteritems():
    setattr(l.config, key, value)

  #config= {key: getattr(l.config,key) for key in l.config.__slots__}
  #l.config_log.append(config)
  #SaveYAML(l.config_log, l.logdir+'config_log.yaml', interactive=False)

def LoadActions(database, i_episode=0, i_node=0):
  xs= LoadYAML(database)['Entry'][i_episode]['Seq'][i_node]['XS']
  act_keys= (
    'gh_ratio','p_pour_trg0','p_pour_trg',
    'dtheta1','dtheta2','shake_spd','shake_range','shake_angle','skill',)
  # act_keys= (
  #   'gh_ratio','p_pour_trg0','p_pour_trg',
  #   'dtheta1','dtheta2')
  # act_keys= (
  #   'gh_ratio','p_pour_trg0','p_pour_trg',
  #   'dtheta1','shake_spd','shake_axis2')
  actions= [ToList(np.array(xs[key]['X']).ravel()) for key in act_keys]
  return {key:(value[0] if len(value)==1 else value) for key,value in zip(act_keys,actions)}

def Run(ct,*args):
  # log_dir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/" \
  #           + "replay/mtr_sms_sv/learn/shake_A/nobounce/002/"
  # target_dir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" \
  #             + "bottomup/learn7/std_pour/ketchup/random/nn_reward/first"+"/"
  target_dir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" \
              + "curriculum/manual_skill_ordering2/ketchup_0055/third"+"/"
  i_episode_list = [26]
  i_node = 0
  n_roop = 1

  for i_episode in i_episode_list:
    for roop in range(n_roop):
      l= TContainer(debug=True)
      l.opt_conf= {}
      l.opt_conf['config']= LoadYAML(target_dir+"config_log.yaml")[i_episode]
      # l.opt_conf['config']= LoadYAML(target_dir+"config_log.yaml")[0]
      # data = LoadYAML(target_dir+"database.yaml")['Entry'][i_episode]['Seq'][0]['XS']
      # l.opt_conf['config'].update({
      #   # "ContactBounce": data["material2"]["X"][0][0], 
      #   # "ContactBounceVel": data["material2"]["X"][1][0],
      #   "ViscosityParam1": data["material2"]["X"][2][0],    #trouble!!!
      #   # "ViscosityMaxDist": data["material2"]["X"][3][0],
      #   "SrcSize2H": data["size_srcmouth"]["X"][0][0]
      # })
      l.opt_conf['actions']= LoadActions(target_dir+'database.yaml',i_episode,i_node)
      l.config_log= []
      l.config_callback= TestConfigCallback
      # ct.Run('mysim.setup.setup2', l)

      ct.Run('mysim.setup.setup_sv', l)

      sim= ct.sim
      l= ct.sim_local
      l.spilled_stop = 10

      obs_keys0= ('ps_rcv','p_pour','p_pour_z','lp_pour','a_trg','size_srcmouth','material2')
      obs_keys_after_grab= obs_keys0+('gh_abs',)
      obs_keys_before_flow= obs_keys_after_grab+('a_pour','a_spill2','a_total')
      obs_keys_after_flow= obs_keys_before_flow+('lp_flow2','lpp_flow','flow_var','da_pour','da_spill2','da_total')
      XS= []

      try:
        XS.append(ObserveXSSA(l,None,obs_keys0+('da_trg',)))
        
        ct.Run('mysim.act.grab_sv', l.opt_conf['actions'])
        XS.append(ObserveXSSA(l,XS[-1],obs_keys_after_grab))

        ct.Run('mysim.act.move_to_rcv_sv', l.opt_conf['actions'])
        XS.append(ObserveXSSA(l,XS[-1],obs_keys_after_grab+('dps_rcv','v_rcv','da_trg')))
        
        ct.Run('mysim.act.move_to_pour_sv', l.opt_conf['actions'])
        XS.append(ObserveXSSA(l,XS[-1],obs_keys_before_flow))

        # ct.Run('mysim.act.shake_A_5s_sv', l.opt_conf['actions'])
        # XS.append(ObserveXSSA(l,XS[-1],obs_keys_after_flow))

        # ct.Run('mysim.act.std_pour_sv_custom', l.opt_conf['actions'])
        # # ct.Run('mysim.act.std_pour_sv_prev', l.opt_conf['actions'])
        # XS.append(ObserveXSSA(l,XS[-1],obs_keys_after_flow))

        if l.opt_conf['actions']['skill']==0:
          ct.Run('mysim.act.std_pour_sv_custom', l.opt_conf['actions'])
        else:
          l.opt_conf['actions']
          actions = {"dtheta1": l.opt_conf['actions']["dtheta1"], "shake_spd": l.opt_conf['actions']["shake_spd"], "shake_axis2": ToList([l.opt_conf['actions']['shake_range'], l.opt_conf['actions']['shake_angle']])}
          ct.Run('mysim.act.shake_A_5s_sv', actions)
        XS.append(ObserveXSSA(l,XS[-1],obs_keys_after_flow))

        # SaveYAML(XS,log_dir+"_ep"+str(i_episode)+'_%s.dat'%TimeStr('short2'))
        Print(XS)

      finally:
        sim.StopPubSub(ct,l)
        l.sensor_callback= None
        ct.srvp.ode_pause()
