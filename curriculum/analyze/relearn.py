# from core_tool import *
from util import *
from ..tasks_domain.detach_datotal.pouring import task_domain as td


def Help():
    pass


def Run(ct, *args):
    model_path = "curriculum5/c1/t5/g5"
    save_path = "curriculum5/c1/t5/relearn/Ftipamount_test"+str(args[0])
    model_name = "Ftip_amount"
    pref = ""
    
    model, DataX, DataY = remake_model(td, model_name, model_path, save_path, pref, is_prev_model=True)
    print(model.Options)
    for x,y in zip(DataX, DataY):
        model.Update(x.tolist(),y.tolist(),not_learn=False)
    
    check_or_create_dir(ROOT_PATH+save_path)
    with open(ROOT_PATH+save_path+"/{}_{}.pkl".format(model_name, pref), "wb") as f:
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