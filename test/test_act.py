#!/usr/bin/python
from core_tool import *
def Help():
  return '''Test of using actions for ODE grasping and pouring simulation (ver.2.1).
    Based on tsim.sm4
  Usage: tsim2.test_act'''

def TestConfigCallback(ct,l,sim):
  # l.config.SrcSize2H= 0.12
  #l.config.RcvPos= [0.6, l.config.RcvPos[1], l.config.RcvPos[2]]
  l.config.RcvPos= [0.8, l.config.RcvPos[1], l.config.RcvPos[2]]
  #l.config.RcvPos= [1.2, l.config.RcvPos[1], l.config.RcvPos[2]]
  #l.config.RcvPos= [0.8+0.6*(random.random()-0.5), l.config.RcvPos[1], l.config.RcvPos[2]]
  CPrint(3,'l.config.RcvPos=',l.config.RcvPos)
  #l.config.ContactBounce= 0.7
  #l.config.ContactBounce= 0.1+0.8*(random.random())
  #CPrint(3,'l.config.ContactBounce=',l.config.ContactBounce)

def Run(ct,*args):
  l= TContainer(debug=True)
  l.config_callback= TestConfigCallback
  ct.Run('mysim.setup', l)
  sim= ct.sim
  l= ct.sim_local

  try:
    # ct.Run('mysim.act.grab', {'gh_ratio':0.5})
    # ct.Run('mysim.act.move_to_rcv', {'p_pour_trg0':[l.sensors.x_rcv.position.x-0.1, l.sensors.x_rcv.position.z+0.4+0.2]})
    # ct.Run('mysim.act.move_to_pour', {'p_pour_trg':[l.sensors.x_rcv.position.x-0.1, l.sensors.x_rcv.position.z+0.4]})

    ct.Run('mysim.act.move_to_rcv', {'p_pour_trg0':[l.sensors.x_src.position.x+0.2, l.sensors.x_src.position.z+0.2]})
    # ct.Run('mysim.act.mypour', {'dtheta1':Rand(0.01, 0.03), 'dtheta2':Rand(0.005, 0.03)})
    # ct.Run('mysim.act.myshake', {'dtheta1':Rand(0.01, 0.03), 'shake_spd':Rand(0.5, 1.0), 'shake_axis2':[Rand(0.05,0.1), Rand(-0.5*math.pi,0.5*math.pi)]})
    ct.Run('mysim.act.shake_B', {'dtheta1':Rand(0.01, 0.03), 'shake_spd_B':Rand(2.0, 8.0), "shake_range":Rand(0.02, 0.06)})
    # ct.Run('mysim.test_spline', {})

    # if RandI(1)==0:
    #   ct.Run('mysim.act.std_pour', {'dtheta1':Rand(0.01, 0.03), 'dtheta2':Rand(0.005, 0.03)})
    # else:
    #   ct.Run('mysim.act.shake_A', {'dtheta1':Rand(0.01, 0.03), 'shake_spd':Rand(0.5, 1.0), 'shake_axis2':[Rand(0.05,0.1), Rand(-0.5*math.pi,0.5*math.pi)]})

  finally:
    sim.StopPubSub(ct,l)
    l.sensor_callback= None
    ct.srvp.ode_pause()
