from core_tool import *
from util import *


def Help():
    pass


def Run(ct, *args):
    model_path_list = [
        "curriculum3/scaling/full_scratch/t1/first100",
        "curriculum3/scaling/full_scratch/t1/second200"
        # "curriculum5/c1/t5/g0",
        # "curriculum5/c1/t5/g1",
        # "curriculum5/c1/t5/g2",
        # "curriculum5/c1/t5/g3",
        # "curriculum5/c1/t5/g4",
        # "curriculum5/c1/t5/g5",
        # "curriculum4/c2/subtask/t1/first200"
    ]
    save_path = "curriculum3/scaling/full_scratch/t1"
    save_img_dir = PICTURE_DIR + save_path.replace("/","_") + "/"
    model_name = "Ftip_amount"
    suff = ""
    
    train_dir_list = [ROOT_PATH + model_path + "/models/train" + "/" for model_path in model_path_list]
    all_nn_log_files = sorted([nn_log_file for train_dir in train_dir_list for nn_log_file in glob.glob(train_dir+"*") if ((model_name in nn_log_file) and ("nn_log-" in nn_log_file))])
    
    # # plt.close()
    # fig = plt.figure(figsize=(15,10))
    # fig.suptitle(model_name+"\n"+save_path)
    # for i, model_type in enumerate(MODEL_TYPES):
    #     nn_log_files = [nn_log_file for nn_log_file in all_nn_log_files if model_type in nn_log_file]
    #     times_list, loss_list, newep_list = [], [], []
    #     for nn_log_file in nn_log_files:
    #         loss_data =[time + times_list[-1] if not len(times_list)==0 else time for time in np.loadtxt(nn_log_file, comments='!').transpose()[1]]
    #         times_list += loss_data
    #         loss_list += list(np.loadtxt(nn_log_file, comments='!').transpose()[2])
    #         newep_list += ([loss_data[0]] + [np.nan]*(len(loss_data)-1))

    #     ax = fig.add_subplot(2,1,i+1)
    #     ax.scatter(times_list, loss_list)
    #     ax.scatter(newep_list, loss_list, c="red")
    #     ax.set_yscale('log')
    #     plt.gca().set_ylim(bottom=1e-4)
    #     ax.set_title("{} model".format(model_type))
    #     ax.set_xlabel("update times (every 50 times)")
    #     ax.set_ylabel("batch loss")
    
    # check_or_create_dir(save_img_dir)
    # plt.savefig(save_img_dir + "batchloss_" + model_name + suff +".png")
    
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=["mean model", "error model"],
        horizontal_spacing = 0.1,
        vertical_spacing = 0.1,
    )
    for i, model_type in enumerate(MODEL_TYPES):
        nn_log_files = [nn_log_file for nn_log_file in all_nn_log_files if model_type in nn_log_file]
        times_list, loss_list, newep_list, count_list = [], [], [], []
        for j, nn_log_file in enumerate(nn_log_files):
            loss_data =[time + times_list[-1] if not len(times_list)==0 else time for time in np.loadtxt(nn_log_file, comments='!').transpose()[1]]
            times_list += loss_data
            loss_list += list(np.loadtxt(nn_log_file, comments='!').transpose()[2])
            newep_list += ([loss_data[0]] + [np.nan]*(len(loss_data)-1))
            count_list += [j]*len(loss_data)
        anno_text = ["loss: {}<br />time: {}<br />count: {}<br />".format(loss,time,count) for loss,time,count in zip(loss_list, times_list, count_list)]
        
        fig.add_trace(go.Scatter(
            x=times_list, y=loss_list, 
            mode='markers',
            marker_color="blue",
            showlegend=False,
            hoverinfo='text',
            text=anno_text,
        ), i+1,1)
        fig.add_trace(go.Scatter(
            x=newep_list, y=loss_list, 
            mode='markers', 
            marker_color="red",
            showlegend=False,
            hoverinfo='text',
            text=anno_text,
        ), i+1,1)
        fig['layout']['xaxis{}'.format(i+1)]['title'] = "update times (every 50 times)"
        fig['layout']['yaxis{}'.format(i+1)]['title'] = "batch loss"
        
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "batchloss_" + model_name + suff +".html", auto_open=False)