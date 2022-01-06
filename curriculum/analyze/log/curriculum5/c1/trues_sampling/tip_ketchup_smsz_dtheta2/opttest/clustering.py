from logging import root
from mode_edge2 import *
from scipy import cluster
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ward, to_tree, leaves_list
from scipy.spatial.distance import pdist
import sys


def read_data(dm):
    log = dm.log
    Y_tip = []
    Y_shake = []
    for i in log['ep']:
        r = log['r_at_est_optparam'][i]
        if log['skill'][i] == 'tip':
            Y_tip.append(r)
        else:
            Y_shake.append(r)
            
    return Y_tip, Y_shake


def hierarchy(Y, dist_thr):
    df = pd.DataFrame(Y)
    Y = np.array(Y)
    Z = linkage(df, method="complete", metric="euclidean")
    
    def get_leafnode_ids(rootnode):
        reafnodes = []
        chek_node_list = [rootnode]
        while len(chek_node_list) > 0:
            node = chek_node_list.pop(0)
            if node.is_leaf():
                reafnodes.append(node.id)
            else:
                left, right = node.left, node.right
                chek_node_list.append(left)
                chek_node_list.append(right)
                
        return reafnodes
    
    rootnode, nodelist = to_tree(Z, rd=True)
    chek_rootnode_list = [rootnode]
    under_distthr_realfnode_ids_list = []
    cluster_rootnode_list = []
    cluster_reafnode_ids_list = []
    while len(chek_rootnode_list) > 0:
        rootnode = chek_rootnode_list.pop(0)
        left, right = rootnode.left, rootnode.right
        for node in [left,right]:
            if node.dist >= dist_thr:
                chek_rootnode_list.append(node)
            else:
                reafnode_ids = get_leafnode_ids(node)
                under_distthr_realfnode_ids_list.append(reafnode_ids)
        if left.dist < dist_thr and right.dist < dist_thr:
            cluster_rootnode_list.append(rootnode)
            reafnode_ids = get_leafnode_ids(rootnode)
            cluster_reafnode_ids_list.append(reafnode_ids)
            
    cluster_mean_value_list = [np.mean(Y[reafnode_ids]) for reafnode_ids in cluster_reafnode_ids_list]
    for under_distthr_realfnode_ids in under_distthr_realfnode_ids_list:
        min_diff = sys.maxsize
        min_diff_cluster_id = None
        for i, cluster_mean_value in enumerate(cluster_mean_value_list):
            diff = abs(np.mean(Y[under_distthr_realfnode_ids]) - cluster_mean_value)
            if diff < min_diff:
                min_diff = diff
                min_diff_cluster_id = i
        cluster_reafnode_ids_list[min_diff_cluster_id] += under_distthr_realfnode_ids
    
    return cluster_reafnode_ids_list, Z


def read_dm(name,i,ch,ver=2):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    logdir = '{}/t{}/{}/'.format(name,i,ch)
    logdir = basedir + logdir
    dm_path = logdir + 'dm.pickle'
    with open(dm_path, mode="rb") as f:
        dm = dill.load(f)

    return dm


def cluseter_plot(Y,clusters):
    color_set = ["blue","red","green","black","purple", "orange"]
    colors = [""]*len(Y)
    for i,cluster in enumerate(clusters):
        color = color_set[i]
        for idx in cluster:
            colors[idx] = color
            
    colors = np.array(colors)[np.argsort(Y)]
    
    fig = plt.figure()
    plt.scatter(x=range(len(Y)), y=sorted(Y), color = colors)
    plt.show()


def hierarchy_plot(Y, Z):   
    fig = plt.figure()
    label = [str(y) for y in Y]
    dendrogram(Z, labels=label)
    plt.show()
    

def set_data(name,i,ch,ver,dist_thr):
    dm = read_dm(name,i,ch,ver)
    Y_tip, Y_shake = read_data(dm)
    clusters_tip, Z_tip = hierarchy(Y_tip, dist_thr)
    clusters_shake, Z_shake = hierarchy(Y_shake, dist_thr)
    return Y_tip,Y_shake,clusters_tip,Z_tip,clusters_shake,Z_shake


def convert_to_cluster(Y, dist_thr):
    Y_cluster = [0]*len(Y)
    clusters, Z = hierarchy(Y, dist_thr)
    for c,cluster in enumerate(clusters):
        for idx in cluster:
            Y_cluster[idx] = c
    return Y_cluster


def Run(ct, *args):
    dist_thr = 0.8
    
    # for i in range(1,99):
    #     Y_tip,Y_shake,clusters_tip,Z_tip,clusters_shake,Z_shake = set_data("GMM12Sig8LCB4/checkpoints",i,"ch500",2, dist_thr = dist_thr)
    #     Print("Index",i,"Tip:",len(clusters_tip),"Shake:",len(clusters_shake))

    i = 98
    Y_tip,Y_shake,clusters_tip,Z_tip,clusters_shake,Z_shake = set_data("GMM12Sig8LCB4/checkpoints",i,"ch500",2, dist_thr = dist_thr)    
    hierarchy_plot(Y_tip, Z_tip)
    cluseter_plot(Y_tip, clusters_tip)
    # hierarchy_plot(Y_shake, Z_shake)
    # cluseter_plot(Y_shake, clusters_shake)