#!/usr/bin/python
from core_tool import *
SmartImportReload('mysim.sm.mysm')
from mysim.sm.mysm import (
  SetMaterial,
  ResetFilter, 
  # ApplyFilter,
  )
def Help():
  return '''Reset command for ODE grasping and pouring simulation.
  Usage:
    tsim2.setup
    tsim2.setup CONFIG_CALLBACK
      CONFIG_CALLBACK: Callback function called after configuration
        (before resetting the simulator).'''

def ApplyFilter(ct,l,sim):
  if 'num_bounce' not in l.filtered:  l.filtered.num_bounce= l.sensors.num_bounce
  else:  l.filtered.num_bounce= max(l.filtered.num_bounce, l.sensors.num_bounce)
  #l.filtered.amount= l.sensors.z_rcv*5.0 if l.sensors.z_rcv>0.0 else 0.0
  # l.filtered.amount= 0.0055*l.sensors.num_rcv

  # if 'v_rcv' not in l.filtered:
  #   l.filtered.v_rcv= [0.0, 0.0, 0.0]
  #   l.filtered.p_rcv_prev= [l.sensors.x_rcv.position.x, l.sensors.x_rcv.position.y, l.sensors.x_rcv.position.z]
  # p_rcv= [l.sensors.x_rcv.position.x, l.sensors.x_rcv.position.y, l.sensors.x_rcv.position.z]
  # l.filtered.v_rcv= [(p_rcv[d]-l.filtered.p_rcv_prev[d])/l.config.TimeStep for d in range(3)]

  ## Ball status kinds (1:in src, 2:in rcv, 3:flow, 4:spill, 0:unknown)
  #if 'ball_st' not in l.filtered:
    ##If only st==1 (in src) is in l.sensors.ball_st:
    #if [st for st in l.sensors.ball_st if st!=1]==[]:
      #l.filtered.ball_st= [1]*len(l.sensors.ball_st)
  #else:

  if l.flow_controlling:
    if 'term_flow_x' not in l.filtered:
      l.filtered.term_flow_x= []
      l.filtered.term_flow_center= [0.0,0.0,0.0]
      l.filtered.term_flow_var= 0.0
      l.filtered.term_flow_max_dist= 0.0
      l.filtered.obs_counter= 0
    #l.sensors.ball_st, l.sensors.ball_x
    flow_x= [l.sensors.ball_x[6*i:6*(i+1)] for i in range(len(l.sensors.ball_st)) if l.sensors.ball_st[i]==3]
    #Note: ball_st[i]==3 denotes a flow particle
    #flow_x.sort(key=lambda x:x[2])  #sort by z
    #N= 10
    #term_flow_x= flow_x[:N]  #particles on the floor or the bottom of the receiver
    l.filtered.term_flow_x+= [xf for xf in flow_x if xf[2]<0.1]  #particles whose height is < 10 cm
    ###
    # random.shuffle(l.filtered.term_flow_x)
    # N= 50
    # l.filtered.term_flow_x= l.filtered.term_flow_x[:N]
    ###
    term_flow_x= l.filtered.term_flow_x
    N= len(term_flow_x)
    if N>0:
      #term_flow_x.sort(key=lambda x:x[2])  #sort by z
      term_flow_center= [sum([term_flow_x[i][0] for i in range(N)])/float(N),
                        sum([term_flow_x[i][1] for i in range(N)])/float(N),
                        #sum([term_flow_x[i][2] for i in range(N)])/float(N)
                        0.0 ]
      dist_x_data= [(Dist(term_flow_x[i][:3], term_flow_center), term_flow_x[i]) for i in range(N)]
      dist_x_data.sort(reverse=True, key=lambda x:x[0])
      term_flow_var= sum([d*d for d,x in dist_x_data]) / float(N)
      term_flow_max_dist= dist_x_data[0][0]
      #term_flow_center : center of flow
      #dist_x_data[0][0] : most far flow
      #dist_x_data[0][1] : corresponding point
      l.filtered.obs_counter+= 1
      if l.filtered.term_flow_center==[0.0,0.0,0.0]:
        l.filtered.term_flow_center= term_flow_center
      else:
        alpha= 0.5/(0.1*l.filtered.obs_counter+1.0)
        l.filtered.term_flow_center= [(1.0-alpha)*f0+alpha*f1 for f0,f1 in zip(l.filtered.term_flow_center,term_flow_center)]
      l.filtered.term_flow_var= min(max(l.filtered.term_flow_var, term_flow_var), 1.0)
      l.filtered.term_flow_max_dist= min(max(l.filtered.term_flow_max_dist, term_flow_max_dist), 1.0)

  #Visualize flow:
  if 'term_flow_max_dist' in l.filtered and l.filtered.term_flow_max_dist>0.0:
    msg= ay_sim_msgs.msg.ODEViz()
    if 'user_viz' in l:
      msg.objects+= l.user_viz
    prm= ay_sim_msgs.msg.ODEVizPrimitive()
    prm.type= prm.LINE
    prm.pose.position.x= l.filtered.term_flow_center[0]
    prm.pose.position.y= l.filtered.term_flow_center[1]
    prm.pose.position.z= l.filtered.term_flow_center[2]
    prm.param= [0.0,0.0,0.3]
    prm.color.r= 0.0
    prm.color.g= 1.0
    prm.color.b= 1.0
    prm.color.a= 0.2
    msg.objects.append(prm)
    prm= ay_sim_msgs.msg.ODEVizPrimitive()
    prm.type= prm.CYLINDER
    prm.pose.position.x= l.filtered.term_flow_center[0]
    prm.pose.position.y= l.filtered.term_flow_center[1]
    prm.pose.position.z= l.filtered.term_flow_center[2]
    prm.param= [math.sqrt(l.filtered.term_flow_var), 0.015]
    prm.color.r= 1.0
    prm.color.g= 1.0
    prm.color.b= 0.0
    prm.color.a= 0.1
    msg.objects.append(prm)
    prm= ay_sim_msgs.msg.ODEVizPrimitive()
    prm.type= prm.CYLINDER
    prm.pose.position.x= l.filtered.term_flow_center[0]
    prm.pose.position.y= l.filtered.term_flow_center[1]
    prm.pose.position.z= l.filtered.term_flow_center[2]
    prm.param= [l.filtered.term_flow_max_dist, 0.007]
    prm.color.r= 1.0
    prm.color.g= 1.0
    prm.color.b= 0.0
    prm.color.a= 0.1
    msg.objects.append(prm)
    ct.pub.ode_viz.publish(msg)
  elif 'user_viz' in l:
    msg= ay_sim_msgs.msg.ODEViz()
    msg.objects+= l.user_viz
    ct.pub.ode_viz.publish(msg)

def Run(ct,*args):
  #config_callback= args[0] if len(args)>0 else None
  sim_local= args[0] if len(args)>0 else None

  #sim= ct.sim if 'sim' in ct.__dict__ else ct.Load('tsim.core1')
  sim= ct.Load('mysim.core.core_sv')
  #l= ct.sim_local if 'sim_local' in ct.__dict__ else TContainer(debug=True)
  l= sim_local if sim_local is not None else TContainer(debug=True)
  ct.sim= sim
  ct.sim_local= l

  #l.planlearn_callback= TestPlanLearnCallback
  #l.config_callback= config_callback

  sim.SetupServiceProxy(ct,l)
  sim.SetupPubSub(ct,l)

  l.max_duration= 500.0
  l.amount_trg= 0.3
  # l.spilled_stop= 5

  #NOTE: These are used in actions (tsim2.act.*)
  l.IsTimeout= lambda: (l.sensors.time-l.start_time > l.max_duration)
  # l.IsPoured= lambda: (l.filtered.amount > l.amount_trg)
  l.IsEmpty= lambda: (l.sensors.num_src <= 20)
  # l.IsSpilled= lambda: (l.filtered.dnum_spill >= l.spilled_stop)
  #l.IsSpilled= lambda: (Print('dnum_spill=',l.filtered.dnum_spill,l.spilled_stop,(l.filtered.dnum_spill >= l.spilled_stop)),
                        #(l.filtered.dnum_spill >= l.spilled_stop))[-1]
  # l.IsSpilledEmpty= lambda: l.IsSpilled() or l.IsEmpty()
  l.IsFlowedOut = lambda: (0.0055*(l.config.BallNum - l.sensors.num_src) >= l.amount_trg)

  ct.srvp.ode_resume()
  # ct.srvp.ode_pause()
  # CPrint(1,ct.sim_local)
  # time.sleep(0.5)
  # for i in range(1):
  #   # ct.srvp.ode_get_config()
  #   CPrint(2,"ct.sim_local.sensors.x_rcv.position.x:",ct.sim_local.sensors.x_rcv.position.x)
  l.config= sim.GetConfig(ct)
  # print(ct.sim_local.sensors)
  # CPrint(1,ct.sim_local)
  print 'Previous config:\n',l.config

  #Setup config
  l.config.BallNum = 100
  l.config.MaxContacts= 2
  l.config.TimeStep= 0.025
  l.config.Gravity= -1.0
  l.config.BallType= 0  #Sphere particles
  #l.config.BallType= 1  #Box particles
  l.config.SrcSize2H= 0.08  #Mouth size; Default: 0.03
  SetMaterial(l, preset='ketchup')
  if 'config_callback' in l and l.config_callback!=None:
    l.config_callback(ct,l,sim)
  #Log('After l.config_callback')
  print 'Config:\n',l.config

  #Reset to get state for plan
  sim.ResetConfig(ct,l.config)


  ###
  time.sleep(0.1)  #Wait for l.sensors is updated
  # time.sleep(5.0)  #Wait for l.sensors is updated
  ###


  #ct.srvp.ode_pause()  #Pause to wait grasp plan

  ResetFilter(l)
  l.flow_controlling= False
  l.sensor_callback= lambda:ApplyFilter(ct,l,sim)  #Activate filter

  sim.GetSensor(ct,l)
  l.start_time= l.sensors.time