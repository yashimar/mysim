import matplotlib.pyplot as plt
import numpy as np
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  file_name = "setup_param_mc10_erp3"
  path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/debug/rcv_logs/" \
          + file_name + ".txt"
  with open(path) as f:
    data = f.readlines()
  num_block = len(data)/10

  i_px = 1; i_py = 2; i_pz = 3
  i_rx = 5; i_ry = 6; i_rz = 7; i_rw = 8
  px, py, pz = [], [], []
  rx, ry, rz, rw = [] ,[], [], []
  for n in range(num_block):
    px.append(float(data[n*10+i_px].split(" ")[-1]))
    py.append(float(data[n*10+i_py].split(" ")[-1]))
    pz.append(float(data[n*10+i_pz].split(" ")[-1]))
    rx.append(float(data[n*10+i_rx].split(" ")[-1]))
    ry.append(float(data[n*10+i_ry].split(" ")[-1]))
    rz.append(float(data[n*10+i_rz].split(" ")[-1]))
    rw.append(float(data[n*10+i_rw].split(" ")[-1]))
  px, py, pz = np.array(px), np.array(py), np.array(pz)
  rx, ry, rz, rw = np.array(rx), np.array(ry), np.array(rz), np.array(rw)
  

  fig = plt.figure(figsize=(15,8))
  fig.suptitle("sensor x_rcv value" + " (MaxContacts 10, ContactSoftCFM 1e-1, ContactSoftERP 0.3)")
  obs_list = [px,py,pz,None,rx,ry,rz,rw]
  ideal_list = [0.6+0,0+0,0+0.5*0.005,None,0,0,0,1]
  name_list = ["position x","position y","position z",None,
              "orientation x","orientation y","orientation z","orientation w"]
  for i, obs in enumerate(obs_list):
    if i==3: continue
    ax = fig.add_subplot(2,len(obs_list)/2,i+1)
    ax.set_title(name_list[i],y=1.07)
    ax.plot(np.arange(1,len(obs)+1),obs,label="observe")
    ax.plot(np.arange(1,len(obs)+1),[ideal_list[i]]*len(obs),c="pink",label="ideal")
    # ax.set_xticks([1e0,1e1,1e2,1e3,1e4])
    ax.set_xscale("log")
    # ax.legend()
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
  plt.subplots_adjust(left=0.105,right=0.92,bottom=0.1,top=0.87)

  plt.savefig("/home/yashima/Pictures/debug/"+file_name+".png")