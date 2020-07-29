import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

def main():
  data_dir = "merged_data3"
  dynamics_dict = {
    'Fgrasp': 10,
    'Fmvtorcv_rcvmv': 10,
    'Fmvtorcv': 10,
    'Fmvtopour2': 10,
    'Fflowc_tip10': 10,
    'Fflowc_shakeA10': 10,
    'Famount4': 10
  }
  root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/data/"
  data_path = root_path + data_dir

  fig = plt.figure(figsize=(14,5))
  fig.suptitle("kmeans")
  plt.subplots_adjust(wspace=0.4,hspace=0.4)
  for i, dynamics in enumerate(dynamics_dict.keys()):
    with open(data_path+"/"+dynamics+"_training_data.pickle", mode='r') as fp:
      data = pickle.load(fp)
    
    n_clusters = dynamics_dict[dynamics]
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=10).fit(data)
    labels = kmeans_model.labels_
    c = Counter(labels)
    m = c.most_common()
    for j in range(len(labels)):
      for k,m_tuple in enumerate(m):
        if m_tuple[0]==labels[j]:
          labels[j] = k
          break

    ax = fig.add_subplot(2,int(len(dynamics_dict)/2)+1,i+1)
    ax.set_title(dynamics+" DataX clusters ({})".format(n_clusters))
    ax.hist(labels)
  plt.show()

if __name__ == "__main__":
  main()