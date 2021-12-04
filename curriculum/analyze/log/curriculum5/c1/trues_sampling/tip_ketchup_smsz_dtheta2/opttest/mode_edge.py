from learn3 import *
import cv2
from scipy import interpolate


def detect_edge(name, ver = 2):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    # name = "GMMSig5LCB3/t1"
    logdir = basedir + "{}/".format(name)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/".format(ver, name)
    check_or_create_dir(save_img_dir)
    with open(logdir+"log.yaml", "r") as yml:
        log = yaml.load(yml)
    dm = Domain3.load(logdir+"dm.pickle")
    datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
    
    eval_tip = evaluation[TIP]
    eval_tip = np.maximum(eval_tip, -3)
    
    kernel_x = np.array([
        [0,-1,0],
        [0,0,0],
        [0,1,0]
    ])
    kernel_y = np.array([
        [0,0,0],
        [-1,0,1],
        [0,0,0]
    ])
    eval_tip_edge = np.sqrt(cv2.filter2D(eval_tip, -1, kernel_x)**2 + cv2.filter2D(eval_tip, -1, kernel_y)**2)
    eval_tip_edge = (eval_tip_edge - eval_tip_edge.min()) / (eval_tip_edge.max() - eval_tip_edge.min())
    
    y = np.max(eval_tip_edge, axis=0)
    fitted_curve = interpolate.interp1d(dm.smsz, y, kind='cubic')
    
    x = np.linspace(0.3,0.8,1000)
    fy = np.maximum(fitted_curve(x), 0)
    
    fig = plt.figure()
    plt.scatter(dm.smsz, fitted_curve(dm.smsz))
    plt.plot(dm.smsz, y)
    plt.show()
    
    fig = plt.figure()
    plt.hist(np.random.choice(x, size=(10000), p=fy/np.sum(fy)), bins=40)
    plt.show()
        
    # fig = go.Figure()
    # fig.add_trace(go.Heatmap(
    #             z = eval_tip_edge, x = dm.smsz, y = dm.dtheta2,
    #             # colorscale = cs if cs != None else "Viridis",
    #             # zmin = zmin, zmax = zmax,
    #             # colorbar=dict(
    #             #     titleside="top", ticks="outside",
    #             #     x = posx_set[col-1], y = posy_set[row-1],
    #             #     thickness=23, len = clength,
    #             #     # tickcolor = "black",
    #             #     tickfont = dict(color = "black"),
    #             # ),
    # ))
    # fig.show()


def Run(ct,*args):
    name = "GMM12Sig8LCB4/checkpoints/t{}/ch100".format(args[0])
    detect_edge(name, ver = 3)