#!/bin/sh
for i in `seq 1`
do
  # ROS_MASTER_URI=http://localhost:11312 && roslaunch -p 11312 my_sim ode_grpour_sim1.launch &
  ROS_MASTER_URI=http://localhost:11312 && roslaunch -p 11312 my_sim ode_grpour_sim_noX.launch &
  sleep 3 && ROS_MASTER_URI=http://localhost:11312 && rosservice call /ode_grpour_sim/resume &
  sleep 20
  # ROS_MASTER_URI=http://localhost:11312 && rosservice call /ode_grpour_sim/pause
  ROS_MASTER_URI=http://localhost:11312 && rosrun ay_trick scripts/direct_run.py mysim.replay.replay2
  killall -9 rosmaster
done