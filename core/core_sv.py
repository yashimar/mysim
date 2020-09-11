#!/usr/bin/python
from core_tool import *
import std_msgs.msg
import std_srvs.srv
roslib.load_manifest('my_sim_msgs')
import my_sim_msgs.msg
import my_sim_msgs.srv
import rospy
def Help():
  return '''Core of ODE grasping and pouring simulation.
  Usage: do not call this directly.'''

def SetupServiceProxy(ct,l):
  if 'ode_get_config' not in ct.srvp:
    print 'Waiting for /ode_grpour_sim2/get_config...'
    rospy.wait_for_service('/ode_grpour_sim2/get_config',3.0)
    ct.srvp.ode_get_config= rospy.ServiceProxy('/ode_grpour_sim2/get_config', my_sim_msgs.srv.ODEGetConfig, persistent=False)
  if 'ode_reset2' not in ct.srvp:
    print 'Waiting for /ode_grpour_sim2/reset2...'
    rospy.wait_for_service('/ode_grpour_sim2/reset2',3.0)
    ct.srvp.ode_reset2= rospy.ServiceProxy('/ode_grpour_sim2/reset2', my_sim_msgs.srv.ODEReset2, persistent=False)
  if 'ode_pause' not in ct.srvp:
    print 'Waiting for /ode_grpour_sim2/pause...'
    rospy.wait_for_service('/ode_grpour_sim2/pause',3.0)
    ct.srvp.ode_pause= rospy.ServiceProxy('/ode_grpour_sim2/pause', std_srvs.srv.Empty, persistent=False)
  if 'ode_resume' not in ct.srvp:
    print 'Waiting for /ode_grpour_sim2/resume...'
    rospy.wait_for_service('/ode_grpour_sim2/resume',3.0)
    ct.srvp.ode_resume= rospy.ServiceProxy('/ode_grpour_sim2/resume', std_srvs.srv.Empty, persistent=False)
  if 'ode_sensors' not in ct.srvp:
    print 'Waiting for /ode_grpour_sim2/get_sensor...'
    rospy.wait_for_service('/ode_grpour_sim2/get_sensor',3.0)
    ct.srvp.ode_get_sensor= rospy.ServiceProxy('/ode_grpour_sim2/get_sensor', my_sim_msgs.srv.ODEGetSensor, persistent=False)
  if 'ode_process_sim' not in ct.srvp:
    print 'Waiting for /ode_grpour_sim2/process_sim...'
    rospy.wait_for_service('/ode_grpour_sim2/process_sim',3.0)
    ct.srvp.ode_process_sim= rospy.ServiceProxy('/ode_grpour_sim2/process_sim', my_sim_msgs.srv.ODEProcessSim, persistent=False)

def SetupPubSub(ct,l):
  StopPubSub(ct,l)
  # if 'ode_ppour' not in ct.pub:
  #   ct.pub.ode_ppour= rospy.Publisher("/ode_grpour_sim2/ppour", std_msgs.msg.Float64MultiArray)
  # if 'ode_theta' not in ct.pub:
  #   ct.pub.ode_theta= rospy.Publisher("/ode_grpour_sim2/theta", std_msgs.msg.Float64)
  if 'ode_viz' not in ct.pub:
    ct.pub.ode_viz= rospy.Publisher("/ode_grpour_sim2/viz", my_sim_msgs.msg.ODEViz)
  # if 'ode_sensors' not in ct.sub:
  #   ct.sub.ode_sensors= rospy.Subscriber("/ode_grpour_sim2/sensors", my_sim_msgs.msg.ODESensor, lambda msg:ODESensorCallback(msg,ct,l))
  # if 'sensor_callback' not in l:
  #   l.sensor_callback= None

def StopPubSub(ct,l):
  if 'ode_sensors' in ct.sub:
    ct.sub.ode_sensors.unregister()
    del ct.sub.ode_sensors

def ODESensorCallback(msg,ct,l):
  l.sensors= msg
  #print l.sensors.num_src, l.sensors.num_rcv, l.sensors.num_flow, l.sensors.num_spill, l.sensors.num_bounce
  if l.sensor_callback!=None:
    l.sensor_callback()

def GetConfig(ct):
  return ct.srvp.ode_get_config().config

def GetSensor(ct,l):
  l.sensors = ct.srvp.ode_get_sensor().sensor
  if l.sensor_callback!=None:
    l.sensor_callback()

def ResetConfig(ct,config):
  ct.pub.ode_viz.publish(my_sim_msgs.msg.ODEViz())  #Clear visualization
  req= my_sim_msgs.srv.ODEReset2Request()
  req.config= config
  ct.srvp.ode_reset2(req)

def SimMove(ct,l,dt,p_pour,theta):
  GetSensor(ct,l)
  p_pour_msg = std_msgs.msg.Float64MultiArray()
  theta_msg = std_msgs.msg.Float64()
  p_pour_msg.data = p_pour
  theta_msg.data = theta

  tc0 = l.sensors.time
  while l.sensors.time-tc0<dt:
    ct.srvp.ode_process_sim(p_pour_msg, theta_msg).success  #process step
    GetSensor(ct,l)

def SimSleep(ct,l,dt):
  GetSensor(ct,l)
  p_pour0 = l.sensors.p_pour
  theta0 = l.sensors.theta
  SimMove(ct,l,dt,p_pour0,theta0)

# def SimSleep(ct,l,dt):
#   tc0= l.sensors.time
#   while l.sensors.time-tc0<dt:
#     time.sleep(dt*0.02)

def MoveDPPour(ct,l,dp):
  GetSensor(ct,l)
  dt= l.config.TimeStep
  p_pour0 = l.sensors.p_pour
  theta0 = l.sensors.theta
  p_pour = Vec(p_pour0) + Vec(dp)
  theta = theta0
  SimMove(ct,l,dt,p_pour,theta)

# def MoveDPPour(ct,l,dp):
#   dt= l.config.TimeStep
#   p_pour0= l.sensors.p_pour
#   p_pour_msg= std_msgs.msg.Float64MultiArray()
#   p_pour_msg.data= Vec(p_pour0) + Vec(dp)
#   ct.pub.ode_ppour.publish(p_pour_msg)
#   SimSleep(ct,l,dt)

def MoveDTheta(ct,l,dth):
  GetSensor(ct,l)
  # dt = l.config.TimeStep
  dt = 1e-4
  p_pour0 = l.sensors.p_pour
  theta0 = l.sensors.theta
  p_pour = Vec(p_pour0)
  theta = theta0 + dth
  SimMove(ct,l,dt,p_pour,theta)

# def MoveDTheta(ct,l,dth):
#   dt= l.config.TimeStep
#   theta0= l.sensors.theta
#   theta_msg= std_msgs.msg.Float64()
#   theta_msg.data= theta0 + dth
#   ct.pub.ode_theta.publish(theta_msg)
#   SimSleep(ct,l,dt)

def MoveDPPourDTheta(ct,l,dp,dth):
  dt= l.config.TimeStep
  p_pour0= l.sensors.p_pour
  p_pour_msg= std_msgs.msg.Float64MultiArray()
  p_pour_msg.data= Vec(p_pour0) + Vec(dp)
  ct.pub.ode_ppour.publish(p_pour_msg)
  theta0= l.sensors.theta
  theta_msg= std_msgs.msg.Float64()
  theta_msg.data= theta0 + dth
  ct.pub.ode_theta.publish(theta_msg)
  SimSleep(ct,l,dt)

def MoveToTrgPPour(ct,l,p_pour_trg,spd):
  GetSensor(ct,l)
  dt= l.config.TimeStep
  p_pour0 = l.sensors.p_pour
  theta0 = l.sensors.theta
  diff = Vec(p_pour_trg) - Vec(p_pour0)
  diff_norm = la.norm(diff)
  if diff_norm>spd*dt:
    p_pour = Vec(p_pour0) + diff*(spd*dt)/diff_norm
    theta = theta0
    reached = False
  else:
    p_pour = Vec(p_pour_trg)
    theta = theta0
    reached = True
  SimMove(ct,l,dt,p_pour,theta)
  return reached

# def MoveToTrgPPour(ct,l,p_pour_trg,spd):
#   dt= l.config.TimeStep
#   p_pour0= l.sensors.p_pour
#   diff= Vec(p_pour_trg) - Vec(p_pour0)
#   diff_norm= la.norm(diff)
#   p_pour_msg= std_msgs.msg.Float64MultiArray()
#   if diff_norm>spd*dt:
#     p_pour_msg.data= Vec(p_pour0) + diff*(spd*dt)/diff_norm
#     reached= False
#   else:
#     p_pour_msg.data= Vec(p_pour_trg)
#     reached= True
#   ct.pub.ode_ppour.publish(p_pour_msg)
#   SimSleep(ct,l,dt)
#   return reached

class TPause:
  def __init__(self, ct):
    self.ct= ct
  def __enter__(self, *args, **kwargs):
    self.ct.srvp.ode_pause()
    return self
  def __exit__(self, *args, **kwargs):
    self.ct.srvp.ode_resume()

def Run(ct,*args):
  print Help()
