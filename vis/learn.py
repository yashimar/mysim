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

  fig = plt.figure(figsize=(20,6))
  fig.suptitle(target_dir)
  for i, dynamics in enumerate(dynamics_list):
    for j, code in enumerate(code_list):
      log_path = root_dir + target_dir + "/models/train/" \
                  + "nn_log-"+dynamics+code+".dat"
      data = np.loadtxt(log_path, comments='!')

      ax = fig.add_subplot(2,len(dynamics_list),i+1+j*len(dynamics_list))
      plt.subplots_adjust(wspace=0.5, hspace=0.6)
      ax.set_title(dynamics+code)
      if y_max!=None: ax.set_ylim(0,y_max)
      else: ax.set_ylim(0,0.01)
      ax.set_xlabel("epochs")
      if j==0: ax.set_ylabel("loss maf (alpha=0.4)")
      else: ax.set_ylabel("loss")
      ax.plot(data[:,0].flatten(),data[:,2].flatten())
  plt.show()