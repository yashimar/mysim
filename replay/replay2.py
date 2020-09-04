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

# def StatObserveXSSA(l,xs_prev,keys,time_step=0.01,n_roop=100):
#   hist = None
#   for i in range(n_roop):
#     time.sleep(time_step)
#     if i==0: 
#       hist = ObserveXSSA(l,xs_prev,keys)
#     else: 
#       obs = ObserveXSSA(l,xs_prev,keys)
#       for key in hist.keys():
#         hist[key].X = np.matrix((np.array(hist[key].X)+np.array(obs[key].X)).tolist())
#   for key in hist.keys(): 
#     hist[key].X = np.matrix((np.array(hist[key].X)/float(n_roop)).tolist())
#   return hist

def TestConfigCallback(ct,l,sim):
  for key,value in l.opt_conf['config'].iteritems():
    setattr(l.config, key, value)

  #config= {key: getattr(l.config,key) for key in l.config.__slots__}
  #l.config_log.append(config)
  #SaveYAML(l.config_log, l.logdir+'config_log.yaml', interactive=False)

def LoadActions(database, i_episode=0, i_node=0):
  xs= LoadYAML(database)['Entry'][i_episode]['Seq'][i_node]['XS']
  # act_keys= (
  #   'gh_ratio','p_pour_trg0','p_pour_trg',
  #   'dtheta1','dtheta2','shake_spd','shake_axis2','skill',)
  # act_keys= (
  #   'gh_ratio','p_pour_trg0','p_pour_trg',
  #   'dtheta1','dtheta2')
  act_keys= (
    'gh_ratio','p_pour_trg0','p_pour_trg',
    'dtheta1','shake_spd','shake_axis2')
  actions= [ToList(np.array(xs[key]['X']).ravel()) for key in act_keys]
  return {key:(value[0] if len(value)==1 else value) for key,value in zip(act_keys,actions)}

def Run(ct,*args):
  log_dir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/debug/" \
            + "replay/shake_A/"
  target_dir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/" \
              + "mtr_sms/infer/additional2_more/shake_A/bounce_009/"
  i_episode_list = [7]
  i_node = 0
  n_roop = 1

  for i_episode in i_episode_list:
    for roop in range(n_roop):
      l= TContainer(debug=True)
      l.opt_conf= {}
      l.opt_conf['config']= LoadYAML(target_dir+"config_log.yaml")[i_episode]
      l.opt_conf['actions']= LoadActions(target_dir+'database.yaml',i_episode,i_node)
      l.config_log= []
      l.config_callback= TestConfigCallback
      # ct.Run('mysim.setup.setup_continue', l)

      Sresume_Fsetup2 = ct.Run('mysim.setup.setup_continue', l)
      start = time.time()

      sim= ct.sim
      l= ct.sim_local

      obs_keys0= ('ps_rcv','p_pour','p_pour_z','lp_pour','a_trg','size_srcmouth','material2')
      obs_keys_after_grab= obs_keys0+('gh_abs',)
      obs_keys_before_flow= obs_keys_after_grab+('a_pour','a_spill2','a_total')
      obs_keys_after_flow= obs_keys_before_flow+('lp_flow2','lpp_flow','flow_var','da_pour','da_spill2','da_total')
      XS= []

      try:
        # Sresume_Sobs0 = time.time()-start+Sresume_Fsetup2
        # CPrint(2,"Start resume to Start observe state0:",Sresume_Sobs0)
        # obs = ObserveXSSA(l,None,obs_keys0+('da_trg',))
        # Sobs0_Fobs0 = time.time()-start+Sresume_Fsetup2-Sresume_Sobs0
        # CPrint(2,"Start observe state0 to Finish observe state0:",Sobs0_Fobs0)
        # Sresume_Fobs0 = Sresume_Sobs0+Sobs0_Fobs0
        # CPrint(2,"Start resume to Finish observe state0:",Sresume_Fobs0)
        # XS.append(obs)
        # obs.update({"Sresume_Fobs0":{"X":[[Sresume_Fobs0]]},
        #             "Sobs0_Fobs0":{"X":[[Sobs0_Fobs0]]}, 
        #             "Sresume_Sobs0":{"X":[[Sresume_Sobs0]]}})
        
        # s1_start = time.time()
        # ct.Run('mysim.act.grab2', l.opt_conf['actions'])
        # Sstate1_Sobs1 = time.time()-s1_start
        # CPrint(2,"Start state1 to Start observe state1:",Sstate1_Sobs1)
        # obs = ObserveXSSA(l,XS[-1],obs_keys_after_grab)
        # Sobs1_Fobs1 = time.time()-s1_start-Sstate1_Sobs1
        # CPrint(2,"Start observe state1 to Finish observe state1:",Sobs1_Fobs1)
        # Sresume_Fobs1 = Sresume_Fobs0+Sstate1_Sobs1+Sobs1_Fobs1
        # CPrint(2,"Start resume to Finish observe state1:",Sresume_Fobs1)
        # XS.append(obs)
        # obs.update({"Sstate1_Sobs1":{"X":[[Sstate1_Sobs1]]},
        #             "Sobs1_Fobs1":{"X":[[Sobs1_Fobs1]]},
        #             "Sresume_Fobs1":{"X":[[Sresume_Fobs1]]}})

        # s2_start = time.time()
        # ct.Run('mysim.act.move_to_rcv2', l.opt_conf['actions'])
        # Sstate2_Sobs2 = time.time()-s2_start
        # CPrint(2,"Start state2 to Start observe state2:",Sstate2_Sobs2)
        # obs = ObserveXSSA(l,XS[-1],obs_keys_after_grab+('dps_rcv','v_rcv','da_trg'))
        # Sobs2_Fobs2 = time.time()-s2_start-Sstate2_Sobs2
        # CPrint(2,"Start observe state2 to Finish observe state2:",Sobs2_Fobs2)
        # Sresume_Fobs2 = Sresume_Fobs1+Sstate2_Sobs2+Sobs2_Fobs2
        # CPrint(2,"Start resume to Finish observe state2:",Sresume_Fobs2)
        # XS.append(obs)
        # obs.update({"Sstate2_Sobs2":{"X":[[Sstate2_Sobs2]]},
        #             "Sobs2_Fobs2":{"X":[[Sobs2_Fobs2]]},
        #             "Sresume_Fobs2":{"X":[[Sresume_Fobs2]]}})
        
        # s3_start = time.time()
        # ct.Run('mysim.act.move_to_pour2', l.opt_conf['actions'])
        # Sstate3_Sobs3 = time.time()-s3_start
        # CPrint(2,"Start state3 to Start observe state3:",Sstate3_Sobs3)
        # obs = ObserveXSSA(l,XS[-1],obs_keys_before_flow)
        # Sobs3_Fobs3 = time.time()-s3_start-Sstate3_Sobs3
        # CPrint(2,"Start observe state3 to Finish observe state3:",Sobs3_Fobs3)
        # Sresume_Fobs3 = Sresume_Fobs2+Sstate3_Sobs3+Sobs3_Fobs3
        # CPrint(2,"Start resume to Finish observe state3:",Sresume_Fobs3)
        # XS.append(obs)
        # obs.update({"Sstate3_Sobs3":{"X":[[Sstate3_Sobs3]]},
        #             "Sobs3_Fobs3":{"X":[[Sobs3_Fobs3]]},
        #             "Sresume_Fobs3":{"X":[[Sresume_Fobs3]]}})

        # ct.srvp.ode_resume()
        time.sleep(0.1)
        obs0 = ObserveXSSA(l,None,obs_keys0+('da_trg',))
        obs1 = ObserveXSSA(l,obs0,obs_keys_after_grab)
        obs2 = ObserveXSSA(l,obs1,obs_keys_after_grab+('dps_rcv','v_rcv','da_trg'))
        obs3 = ObserveXSSA(l,obs2,obs_keys_before_flow)
        # time.sleep(0.1)

        l.theta_init = DegToRad(45.0)
        l.p_pour_trg0 = [0.30012268220115873,0.0,0.552053219530789]
        Sresume_Sstate4 = time.time()-start+Sresume_Fsetup2
        CPrint(2,"Start resume to Start state4:",Sresume_Sstate4)
        s4_start = time.time()
        ct.Run('mysim.act.shake_A_5s2', l.opt_conf['actions'])
        Sstate4_Sobs4 = time.time()-s4_start
        CPrint(2,"Start state4 to Start observe state4:",Sstate4_Sobs4)
        obs = ObserveXSSA(l,obs3,obs_keys_after_flow)
        Sobs4_Fobs4 = time.time()-s4_start-Sstate4_Sobs4
        CPrint(2,"Start observe state4 to Finish observe state4:",Sobs4_Fobs4)
        # Sresume_Fobs4 = Sresume_Fobs3+Sstate4_Sobs4+Sobs4_Fobs4
        # CPrint(2,"Start resume to Finish observe state4:",Sresume_Fobs4)
        obs.update({"Sstate4_Sobs4":{"X":[[Sstate4_Sobs4]]},
                    "Sobs4_Fobs4":{"X":[[Sobs4_Fobs4]]},
                    "Sresume_Sstate4":{"X":[[Sresume_Sstate4]]}})
        XS.append(obs)                    

        # ct.Run('mysim.act.std_pour', l.opt_conf['actions'])
        # ct.Run('mysim.act.shake_A_5s2', l.opt_conf['actions'])
        # # if l.opt_conf['actions']['skill']==0:
        # #   ct.Run('mysim.act.std_pour', l.opt_conf['actions'])
        # # else:
        # #   ct.Run('mysim.act.shake_A_5s', l.opt_conf['actions'])
        # XS.append(ObserveXSSA(l,XS[-1],obs_keys_after_flow))

        SaveYAML(XS,log_dir+"_ep"+str(i_episode)+'_%s.dat'%TimeStr('short2'))

      finally:
        sim.StopPubSub(ct,l)
        l.sensor_callback= None
        ct.srvp.ode_pause()
