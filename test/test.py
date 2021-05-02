#!/usr/bin/python
from core_tool import *
roslib.load_manifest('ay_sim_msgs')
import ay_sim_msgs
print(ay_sim_msgs)
import ay_sim_msgs.msg
import os


def Help():
    pass


def Run(ct, *args):
    print(os.environ['ROS_PACKAGE_PATH'])
