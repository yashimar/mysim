#!/usr/bin/python
from core_tool import *
def Help():
  return '''Primitive action (grasping) for ODE grasping and pouring simulation.
  Usage: Expected to be used from an execution context.'''

def Run(ct,*args):
  params= args[0]  #Parameters (dictionary) of this action

  sim= ct.sim
  l= ct.sim_local

  ##Plan grasp
  #l.planlearn_callback(ct,l,sim,'infer_grab')
  #Log('After infer_grab')

  #gh= ToList(l.xs.n0['gh_ratio'].X)
  #l.config.GripperHeight= gh[0]*l.sensors.p_pour[2]  #Should be in [0,l.sensors.p_pour[2]]
  l.exec_status= SUCCESS_CODE

  gh= params['gh_ratio']
  sim.GetSensor(ct,l)
  l.config.GripperHeight= gh*l.sensors.p_pour[2]  #Should be in [0,l.sensors.p_pour[2]]
  l.exec_status= SUCCESS_CODE

  #ct.srvp.ode_resume()
  #Reset again to apply the grasp plan
  sim.ResetConfig(ct,l.config)


  l.exec_status= SUCCESS_CODE
  sim.GetSensor(ct,l)

  return l.exec_status