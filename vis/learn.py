import matplotlib.pyplot as plt
import numpy as np

def Help():
  pass

def Run(ct,*args):
  target_dir = args[0]
  root_dir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  # root_dir = "/tmp/"
  y_max = args[1] if len(args)==2 else None
  dynamics_list = ["Fgrasp","Fmvtorcv_rcvmv","Fmvtorcv","Fmvtopour2",
                    "Fflowc_tip10","Fflowc_shakeA10","Famount4"]
  code_list = ["mean","err"]

  fig = plt.figure(figsize=(20,12))
  fig.suptitle("target: "+target_dir+
                "\nEMA(0.4) average loss caluculated for each check stop (blue) &" +
                "\nstandard deviation of EMA(0.4) average loss (orange)")
  for i, dynamics in enumerate(dynamics_list):
    for j, code in enumerate(code_list):
      log_path = root_dir + target_dir + "/models/train/" \
                  + "nn_log-"+dynamics+code+".dat"
      data = np.loadtxt(log_path, comments='!')
      for k, stat in enumerate(["mean","sdv"]):
        ax = fig.add_subplot(4,len(dynamics_list),i+1+(2*j+k)*len(dynamics_list))
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        plt.subplots_adjust(wspace=0.7, hspace=0.6)
        ax.set_title(dynamics+" "+code+" model "+stat,fontsize=8)
        if stat=="mean":
          # if y_max!=None: ax.set_ylim(0,y_max)
          # elif dynamics+code=="Fgraspmean": ax.set_ylim(0,0.005)
          # elif dynamics+code=="Fgrasperr": ax.set_ylim(0,0.005)
          # elif dynamics+code=="Fmvtorcv_rcvmvmean": ax.set_ylim(0,0.005)
          # elif dynamics+code=="Fmvtorcv_rcvmverr": ax.set_ylim(0,0.005)
          # elif dynamics+code=="Fmvtorcvmean": ax.set_ylim(0,0.02)
          # elif dynamics+code=="Fmvtorcverr": ax.set_ylim(0,0.005)
          # elif dynamics+code=="Fmvtopour2mean": ax.set_ylim(0,0.02)
          # elif dynamics+code=="Fmvtopour2err": ax.set_ylim(0,0.005)
          # elif dynamics+code=="Fflowc_tip10mean": ax.set_ylim(0,0.02)
          # elif dynamics+code=="Fflowc_tip10err": ax.set_ylim(0,0.02)
          # elif dynamics+code=="Fflowc_shakeA10mean": ax.set_ylim(0,0.02)
          # elif dynamics+code=="Fflowc_shakeA10err": ax.set_ylim(0,0.02)
          # elif dynamics+code=="Famount4mean": ax.set_ylim(0,0.1)
          # elif dynamics+code=="Famount4err": ax.set_ylim(0,0.02)
          # else: ax.set_ylim(0,0.01)
          ax.set_ylabel("EMA loss average",fontsize=8)
          ax.plot(data[:,0].flatten(),data[:,2].flatten())
        elif stat=="sdv": 
          # ax.set_ylim(0,0.01)
          ax.set_ylabel("EMA loss sdv",fontsize=8)
          ax.plot(data[:,0].flatten(),data[:,3].flatten(),c="orange")
        ax.set_xlabel("epochs",fontsize=8)
  plt.show()