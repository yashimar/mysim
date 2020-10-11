#!/usr/bin/python
from core_tool import *
SmartImportReload('mysim.sm.mysm')
from mysim.sm.mysm import (
  SetMaterial,
  ResetFilter, ApplyFilter,
  )
def Help():
  return '''Reset command for ODE grasping and pouring simulation.
  Usage:
    tsim2.setup
    tsim2.setup CONFIG_CALLBACK
      CONFIG_CALLBACK: Callback function called after configuration
        (before resetting the simulator).'''


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
  l.spilled_stop= 5

  #NOTE: These are used in actions (tsim2.act.*)
  l.IsTimeout= lambda: (l.sensors.time-l.start_time > l.max_duration)
  l.IsPoured= lambda: (l.filtered.amount > l.amount_trg)
  l.IsEmpty= lambda: (l.sensors.num_src <= 20)
  l.IsSpilled= lambda: (l.filtered.dnum_spill >= l.spilled_stop)
  #l.IsSpilled= lambda: (Print('dnum_spill=',l.filtered.dnum_spill,l.spilled_stop,(l.filtered.dnum_spill >= l.spilled_stop)),
                        #(l.filtered.dnum_spill >= l.spilled_stop))[-1]
  l.IsSpilledEmpty= lambda: l.IsSpilled() or l.IsEmpty()

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