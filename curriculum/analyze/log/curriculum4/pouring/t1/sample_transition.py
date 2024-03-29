# coding: UTF-8
from core_tool import *
from tsim.dpl_cmn import *
SmartImportReload('tsim.dpl_cmn')
from .....util import *
from ......tasks_domain.adhoc_amp.util import Rmodel
from ......tasks_domain.adhoc_amp.pouring import task_domain as td
AMP_DTHETA2, AMP_SMSZ, AMP_SHAKE_RANGE = td.AMP_DTHETA2, td.AMP_SMSZ, td.AMP_SHAKE_RANGE


def Help():
    pass


def Run(ct, *args):
    log_name_list = [
        "curriculum4/pouring/full_scratch/curriculum_test/t1/first300",
    ]
    save_dir = PICTURE_DIR + "curriculum4/pouring/full_scratch/curriculum_test/t1".replace("/","_") + "/"
    save_sh_dir = "curriculum4/pouring/full_scratch/curriculum_test/t1"
    file_name_pref = ""
    dynamics_iodim_pair = {"Fmvtopour2": (3, 3), "Ftip": (12, 4), "Fshake": (14, 4), "Famount": (12, 2)}
    vis_state_dynamics_outdim_lim_pair = [
        ("lp_pour_x", "Fmvtopour2", 0, (-0.5,0.7)),
        ("lp_pour_z", "Fmvtopour2", 2, (-0.2,0.6)),
        ("da_total_tip", "Ftip", 0, (-0.1,0.9)),
        ("lp_flow_x_tip", "Ftip", 1, (-0.3,1.5)),
        ("lp_flow_y_tip", "Ftip", 2, (-0.1,0.1)),
        ("flow_var_tip", "Ftip", 3, (0.1,1.0)),
        ("da_total_shake", "Fshake", 0, (-0.1,0.9)),
        ("lp_flow_x_shake", "Fshake", 1, (-0.3,1.5)),
        ("lp_flow_y_shake", "Fshake", 2, (-0.1,0.1)),
        ("flow_var_shake", "Fshake", 3, (0.1,1.0)),
        ("da_pour", "Famount", 0, (-0.1,0.9)),
        ("da_spill2", "Famount", 1, (-0.1,7)),
    ]
    
    node_states_dim_pair = [
        ["n0", [("size_srcmouth", 1), ("material2", 4), ("dtheta2", 1), ("shake_spd", 1), ("shake_range", 1), ("shake_angle", 1)]],
        ["n2b", [("lp_pour", 3), ]],
        ["n2c", [("skill", 1), ]],
        ["n3ti", [("da_total", 1), ("lp_flow", 2), ("flow_var", 1)]],
        ["n4ti", [("da_pour", 1), ("da_spill2", 1)]],
        ["n4tir1", [(".r", 1), ]],
        ["n4tir2", [(".r", 1), ]],
        ["n3sa", [("da_total", 1), ("lp_flow", 2), ("flow_var", 1)]],
        ["n4sa", [("da_pour", 1), ("da_spill2", 1)]],
        ["n4sar1", [(".r", 1), ]],
        ["n4sar2", [(".r", 1), ]],
    ]
    
    sh, esh = get_true_and_bestpolicy_est_state_histories(save_sh_dir, log_name_list, node_states_dim_pair, recreate=False)
    num_graph_type = 5
    c_tip = [True if s == 0 else False for s in sh["n2c"]["skill"][MEAN]]
    c_shake = [True if s == 1 else False for s in sh["n2c"]["skill"][MEAN]]
    c_nobounce = [True if m2 == 0.0 else False for m2 in sh["n0"]["material2_2"][MEAN]]
    c_ketchup = [True if m2 == 0.25 else False for m2 in sh["n0"]["material2_2"][MEAN]]
    ct = lambda c1,c2 : [True if b1==b2==True else False for b1,b2 in zip(c1,c2)]
    c_lam_all = lambda j: True
    c_lam_predture = lambda j: True if (j%num_graph_type!=1 and j%num_graph_type!=4) else False
    c_lam_latentpredtrue = lambda j: True if (j%num_graph_type!=0 and j%num_graph_type!=3) else False
    vis_condition_title_pair = sum([[
        ("[{}] full".format(title), [True]*len(sh["n0"]["size_srcmouth"][MEAN]), c_lam),
        ("[{}] Tip".format(title), c_tip, c_lam),
        ("[{}] Tip - nobounce".format(title), ct(c_tip, c_nobounce), c_lam),
        ("[{}] Tip - ketchup".format(title), ct(c_tip, c_ketchup), c_lam),
        ("[{}] Shake".format(title), c_shake, c_lam),
        ("[{}] Shake - nobounce".format(title), ct(c_shake, c_nobounce), c_lam),
        ("[{}] Shake - ketchup".format(title), ct(c_shake, c_ketchup), c_lam),
        ("[{}] nobounce".format(title), c_nobounce, c_lam),
        ("[{}] ketchup".format(title), c_ketchup, c_lam),
    ] for title,c_lam in zip(["All graph","Pred & True","Latent pred & True"],[c_lam_all,c_lam_predture,c_lam_latentpredtrue])], [])
    
    def updatemenu(fig):
        buttons = [{
            'label': title,
            'method': "update",
            'args':[
                    {'visible': [((True if (num_graph_type*i<=j%(num_graph_type*len(vis_condition_title_pair))<num_graph_type*(i+1)) else False) and c_lam(j)) for j in range(num_graph_type*len(vis_condition_title_pair)*len(vis_state_dynamics_outdim_lim_pair))]},
                ]
        } for i, (title, _, c_lam) in enumerate(vis_condition_title_pair)]
        buttons2 =[{
            'label': "test",
            'method': "update",
            'args':[{'visible': [""]*num_graph_type*len(vis_condition_title_pair)*len(vis_state_dynamics_outdim_lim_pair)}]
        }]
        updatemenus = [{
            "type": "dropdown",
            "buttons": buttons,
            "active": 0,
            "x": 0.0,
            "xanchor": 'left',
            "y": 1.005,
            "yanchor": 'top',
        }]
        fig['layout']['updatemenus'] = updatemenus
    
    suff_annotation = lambda df: [
        "<b>policy's estimation</b><br />"
        "　lp_pour xz: ({:.3f}+/-{:.3f},{:.3f}+/-{:.3f})<br />".format(esh["n2b"]["lp_pour_0"][MEAN][i], esh["n2b"]["lp_pour_0"][SIGMA][i], esh["n2b"]["lp_pour_2"][MEAN][i], esh["n2b"]["lp_pour_2"][SIGMA][i])+\
        ("　<b>[tip]</b><br />" if esh["n2c"]["skill"][MEAN][i] == 0 else "　[tip]<br />") +\
        "　　da_total: {:.3f}+/-{:.3f}<br />".format(esh["n3ti"]["da_total"][MEAN][i], esh["n3ti"]["da_total"][SIGMA][i])+\
        "　　lp_flow xy: ({:.3f}+/-{:.3f},{:.3f}+/-{:.3f})<br />".format(esh["n3ti"]["lp_flow_0"][MEAN][i], esh["n3ti"]["lp_flow_0"][SIGMA][i], esh["n3ti"]["lp_flow_1"][MEAN][i], esh["n3ti"]["lp_flow_1"][SIGMA][i])+\
        "　　flow_var: {:.3f}+/-{:.3f}<br />".format(esh["n3ti"]["flow_var"][MEAN][i], esh["n3ti"]["flow_var"][SIGMA][i])+\
        "　　da_pour: {:.3f}+/-{:.3f}<br />".format(esh["n4ti"]["da_pour"][MEAN][i], esh["n4ti"]["da_pour"][SIGMA][i])+\
        "　　da_spill2: {:.3f}+/-{:.3f}<br />".format(esh["n4ti"]["da_spill2"][MEAN][i], esh["n4ti"]["da_spill2"][SIGMA][i])+\
        "　　Rdapour: {:.3f}+/-{:.3f}<br />".format(esh["n4tir1"][".r"][MEAN][i], esh["n4tir1"][".r"][SIGMA][i])+\
        "　　Rdaspill: {:.3f}+/-{:.3f}<br />".format(esh["n4tir2"][".r"][MEAN][i], esh["n4tir2"][".r"][SIGMA][i])+\
        ("　<b>[shake]</b><br />" if esh["n2c"]["skill"][MEAN][i] == 1 else "　[shake]<br />") +\
        "　　da_total: {:.3f}+/-{:.3f}<br />".format(esh["n3sa"]["da_total"][MEAN][i], esh["n3sa"]["da_total"][SIGMA][i])+\
        "　　lp_flow xy: ({:.3f}+/-{:.3f},{:.3f}+/-{:.3f})<br />".format(esh["n3sa"]["lp_flow_0"][MEAN][i], esh["n3sa"]["lp_flow_0"][SIGMA][i], esh["n3sa"]["lp_flow_1"][MEAN][i], esh["n3sa"]["lp_flow_1"][SIGMA][i])+\
        "　　flow_var: {:.3f}+/-{:.3f}<br />".format(esh["n3sa"]["flow_var"][MEAN][i], esh["n3sa"]["flow_var"][SIGMA][i])+\
        "　　da_pour: {:.3f}+/-{:.3f}<br />".format(esh["n4sa"]["da_pour"][MEAN][i], esh["n4sa"]["da_pour"][SIGMA][i])+\
        "　　da_spill2: {:.3f}+/-{:.3f}<br />".format(esh["n4sa"]["da_spill2"][MEAN][i], esh["n4sa"]["da_spill2"][SIGMA][i])+\
        "　　Rdapour: {:.3f}+/-{:.3f}<br />".format(esh["n4sar1"][".r"][MEAN][i], esh["n4sar1"][".r"][SIGMA][i])+\
        "　　Rdaspill: {:.3f}+/-{:.3f}<br />".format(esh["n4sar2"][".r"][MEAN][i], esh["n4sar2"][".r"][SIGMA][i])
    for i in df.index]
    
    go_layout = {
        'height': 800*2*len(vis_state_dynamics_outdim_lim_pair),
        'width': 1800,
        'margin': dict(t=30, b=0, pad=0),
        'legend_tracegroupgap': 800-30,
        'hoverdistance': 5,
    }
    
    transition_plot(td, log_name_list, dynamics_iodim_pair, vis_state_dynamics_outdim_lim_pair, go_layout, suff_annotation, save_dir, file_name_pref, vis_condition_title_pair, updatemenu, is_prev_model=True)