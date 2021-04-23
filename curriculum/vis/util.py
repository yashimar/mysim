from collections import defaultdict
import yaml
import joblib
import numpy as np
import types


def FormatSequence(name_log):
    root_path = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/logs/"
    sl_path = root_path + name_log + "/sequence_list.yaml"
    tree_path = root_path+name_log+"/best_est_trees/"
    # color_list = args[2] if len(args)==3 else None

    with open(sl_path, "r") as yml:
        sl = yaml.load(yml)

    envs = defaultdict(list)
    trues = defaultdict(list)
    skills = []
    for ep in range(len(sl)):
        config = sl[ep]["config"]
        reward = sl[ep]["reward"]
        sequence = sl[ep]["sequence"]

        envs["smsz"].append(config["size_srcmouth"][0][0])
        if config["material2"][0][0] == 0.7:
            envs["mtr"].append("bounce")
        elif config["material2"][2][0] == 0.25:
            envs["mtr"].append("ketchup")
        elif config["material2"][0][0] == 1.5:
            envs["mtr"].append("natto")
        else:
            envs["mtr"].append("nobounce")
        if "sa" in sequence[4].keys()[0]:
            skills.append("shake_A")
        else:
            skills.append("std_pour")
        trues["da_spill2"].append(reward[1][2][0]["da_spill2"])
        trues["da_pour"].append(reward[1][3][0]["da_pour"])
        trues[".r"].append(reward[0]["total"])
        trues["p_pour_trg"].append(sequence[2]["n2a"]["p_pour_trg"])
        trues["lp_pour_x"].append(sequence[3]["n2b"]["lp_pour"][0][0])
        trues["lp_pour_z"].append(sequence[3]["n2b"]["lp_pour"][2][0])
        # trues["da_total"].append(sequence[5][sequence[5].keys()[0]]["da_total"][0][0])

    ests = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i in range(len(sl)):
        with open(tree_path+"ep"+str(i)+"_n0.jb", mode="rb") as f:
            tree = joblib.load(f)
            for n in tree.Tree.keys():
                if n.A == "n4tir":
                    n_tir = n
                elif n.A == "n4sar":
                    n_sar = n
            tir_xs = tree.Tree[n_tir].XS
            sar_xs = tree.Tree[n_sar].XS
            selected_xs = tir_xs if tir_xs[".r"].X.item() > sar_xs[".r"].X.item() else sar_xs
            # for (est_dict, r_xs) in zip([ests["sa"]], [sar_xs]):
            # for (est_dict, r_xs) in zip([ests["ti"]], [tir_xs]):
            for (est_dict, r_xs) in zip([ests["ti"], ests["sa"], ests["selected"]], [tir_xs, sar_xs, selected_xs]):
                # for s in ["da_spill2", "da_pour", ".r"]:
                for s in r_xs.keys():
                    est_dict[s]["mean"].append(r_xs[s].X)
                    if type(r_xs[s].Cov) != types.NoneType:
                        est_dict[s]["sdv"].append(np.sqrt(r_xs[s].Cov))
                # est_dict["skill"]["mean"].append(r_xs["skill"].X.item())
                # est_dict["lp_pour_x"]["mean"].append(r_xs["lp_pour"].X[0].item())
                # est_dict["lp_pour_x"]["sdv"].append(np.sqrt(r_xs["lp_pour"].Cov[0,0].item()))
                # est_dict["lp_pour_z"]["mean"].append(r_xs["lp_pour"].X[2].item())
            # est_dict["lp_pour_z"]["sdv"].append(np.sqrt(r_xs["lp_pour"].Cov[2,2].item()))

    return envs, trues, ests
