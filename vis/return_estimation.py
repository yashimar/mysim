import numpy as np
from matplotlib import pyplot as plt
from core_tool import *

def Help():
  pass

def Run(ct, *args):
  name_log = "mtr_sms_sv/validation/shake_A/natto/0055/1search_nodb"
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
  log_path = root_path + name_log + "/dpl_est.dat"

  data = np.genfromtxt(log_path)
  return_list = data[:,1]
  est_list = data[:,2]
  diff_list = abs(return_list-est_list)
  
  plt.figure()
  plt.title(name_log)
  plt.ylabel("|| actual return - estimation ||")
  plt.boxplot(diff_list, 
              labels=[str(round(np.mean(diff_list),3))+"+/-"
                      +str(round(np.std(diff_list),3))])
  plt.show()