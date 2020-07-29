#!/bin/bash

ids=""
p0=11311
nsim=10
na=1
nb=$((nsim-na))

ROS_DISTR=melodic
. /opt/ros/$ROS_DISTR/setup.bash
. ~/catkin_ws/devel/setup.bash
. /opt/ros/$ROS_DISTR/share/rosbash/rosbash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:$HOME/ros_ws:$HOME/prg/testl/ros_sandbox
export ROBOT=sim
export ROS_IP=127.0.0.1

for p in `seq $p0 $((p0+nsim-1))`;do
  # #No GUI:
  # roslaunch -p $p lfd_sim ode_grpour_sim_noX.launch &
  #With GUI:
  roslaunch -p $p ay_sim ode_grpour_sim1.launch &
  ids="$ids $!"
  sleep 0.1
done

sleep 2

for p in `seq $p0 $((p0+nsim-1))`;do
  ROS_MASTER_URI=http://localhost:$p && rosservice call /ode_grpour_sim/pause
done

for p in `seq $p0 $((p0+na-1))`;do
  # rosrunp $p lfd_trick scripts/cui_tool.py
  ROS_MASTER_URI=http://localhost:$p && rosrun ay_trick scripts/direct_run.py mysim.test_act &
  # ROS_MASTER_URI=http://localhost:$p && rosrun lfd_trick scripts/direct_run.py tsim2.test_pause &
  ids="$ids $!"
done

