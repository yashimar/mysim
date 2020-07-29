import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

def main():
  data_dir = "merged_data3"
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/data/"
  data_path = root_path + data_dir
  dynamics_list = ['Fgrasp','Fmvtorcv_rcvmv','Fmvtorcv','Fmvtopour2',
                    'Fflowc_tip10','Fflowc_shakeA10','Famount4']

  fig = plt.figure(figsize=(20,10))
  fig.suptitle("pca")
  plt.subplots_adjust(wspace=0.4,hspace=0.6)
  for i, dynamics in enumerate(dynamics_list):
    with open(data_path+"/"+dynamics+"_training_data.pickle", mode='r') as fp:
      data = pickle.load(fp)
    
    data -= data.mean()
    data /= data.std()
    pca = PCA()
    pca.fit(data)
    feature = pca.transform(data)
    evr = pca.explained_variance_ratio_

    try:
      ax = fig.add_subplot(4,int(len(dynamics_list)/2)+1,i+1)
      ax.set_title(dynamics+" DataX PC1 PC2")
      ax.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
      ax.set_xlabel("PC1")
      ax.set_ylabel("PC2")

      ax = fig.add_subplot(4,int(len(dynamics_list)/2)+1,i+1+2*(int(len(dynamics_list)/2)+1))
      ax.set_title(dynamics+" PC Cumulative cont rate")
      ax.bar(x=np.linspace(0,len(evr)-1,len(evr)),height=np.cumsum(evr))
      ax.set_xlabel("PC")
      ax.set_ylabel("cumulative rate")
    except:
      pass
  plt.show()

if __name__ == "__main__":
  main()