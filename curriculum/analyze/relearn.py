# from core_tool import *
from util import *
from ..tasks_domain.detach_datotal.pouring import task_domain as td


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum3/scaling/full_scratch/t1/second200"
    save_path = "curriculum3/scaling/full_scratch/t1/relearn/1e32e5"
    model_name = "Famount"
    
    model, DataX, DataY = remake_model(td, model_name, model_path, save_path, is_prev_model=True)
    model.Options.update({
        "batchsize": 10,  # default 10
        "num_max_update": 5000,  # default 5000
        'num_check_stop': 50,  # default 50
        'loss_stddev_stop': 1e-3,  # default 1e-3
        'loss_stddev_stop_err': 2e-5, #default None
        'AdaDelta_rho': 0.9,  # default 0.9
    })
    print(model.Options)
    for i,(x,y) in enumerate(zip(DataX, DataY)):
        # if i>100:
        #     continue
        model.Update(x.tolist(),y.tolist(),not_learn=False)
    
    check_or_create_dir(ROOT_PATH+save_path)
    with open(ROOT_PATH+save_path+"/{}.pkl".format(model_name), "wb") as f:
        pickle.dump(model, f)
        
        
    # with open(ROOT_PATH+"{}/{}_{}.pkl".format(save_path,model_name,pref), "rb") as f:
    #     model = pickle.load(f)
    
    # def pred_test(model):
    #     for i, (x,t) in enumerate(zip(model.DataX, model.DataY)):
    #         p = model.Predict(x, [0.,0.,0.,0.,0.,0.,0.,0.,0.3,0.,0.,0.], with_var=True)
    #         # print("---")
    #         # Print("input:", x)
    #         # Print("est:", p.Y[0].item())
    #         Print(i, "sigma:", np.sqrt(p.Var[0,0]))
    #         # Print("true:", t[0])
    #         # Print("diff:", abs(p.Y[0].item()-t[0]))
        
    # pred_test(model)