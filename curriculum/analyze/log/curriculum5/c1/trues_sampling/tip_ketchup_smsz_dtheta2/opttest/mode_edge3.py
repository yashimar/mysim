#coding: UTF-8
from mode_edge2 import *
from clustering import *


def extract_trainig_data2(dm, dist_thr = 0.8):
    log = dm.log
    X_tip, Y_tip = [], []
    X_shake, Y_shake = [], []
    for i in log['ep']:
        x = []
        r = log['r_at_est_optparam'][i]
        if log['skill'][i] == 'tip':
            x.append(log['est_optparam'][i])
            x.append(log['smsz'][i])
            X_tip.append(x)
            Y_tip.append(r)
        else:
            x.append(log['smsz'][i])
            X_shake.append(x)
            Y_shake.append(r)
            
    Y_tip_org = deepcopy(Y_tip)
    Y_shake_org = deepcopy(Y_shake)
    
    Y_tip = convert_to_cluster(Y_tip, dist_thr)
    Y_shake = convert_to_cluster(Y_shake, dist_thr)
    
    X_tip = torch.tensor(X_tip, dtype = torch.float32)
    Y_tip = torch.tensor(Y_tip, dtype = torch.int64)
    X_shake = torch.tensor(X_shake, dtype = torch.float32)
    Y_shake = torch.tensor(Y_shake, dtype = torch.int64)
    
    return X_tip, Y_tip, X_shake, Y_shake, Y_tip_org, Y_shake_org


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        n_units = 200
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, n_units),
            nn.ReLU(),
            # nn.BatchNorm1d(n_units),
            # nn.Linear(n_units, n_units),
            # nn.ReLU(),
            # nn.BatchNorm1d(n_units),
            nn.Linear(n_units, out_dim),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
    

def train2(X, Y, in_dim):
    out_dim = len(Y.unique())
    model = Classifier(in_dim, out_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model, max_acc = train(model, X, Y, criterion, optimizer, batch_size = len(X), n_epoch = 200)
    # probs, grads, sum_grad = compute_grad(dm, model, out_dim)
    
    return model


def scatter2(X, Y, probs, Y_org, save_img_dir = None):    
    Xc_set = dict()
    out_dim = len(set(Y))
    for c in range(out_dim):
        Xc_set[c] = np.array([x for x, y in zip(X, Y) if y == c])
    
    smsz = np.linspace(0.3,0.8,100)
    dtheta = np.linspace(0.1,1.0,100)[::-1]
    
    f_scatter = lambda X, color, showlegend=False: go.Scatter(
        mode = 'markers', x = X[:,1], y = X[:,0],
        marker = dict(color = color, size = 14, line = dict(color = "black", width = 4)),
        showlegend = showlegend,
    )
    f_heatmap = lambda z, z_range, color_bar_xy, color_scale=None: go.Heatmap(
        z = z, x = smsz, y = dtheta,
        colorscale = color_scale if color_scale != None else "Viridis",
        zmin = z_range[0], zmax = z_range[1],
        colorbar=dict(
            titleside="top", ticks="outside",
            x = color_bar_xy[0], y = color_bar_xy[1],
            thickness=23, len = 0.3,
        ),
    )
    f_update_layout = lambda fig, width, height = 800: fig.update_layout(
        width = width, height = height, 
        xaxis = dict(title = 'smsz', range = (0.28,0.82)), 
        yaxis = dict(title = 'dtheta', range = (0.08,1.02)),
    )
    color_set = ["blue","red","green","black","purple", "orange"]
    
    
    # グラフ1 予測分布プロット
    subplot_titles = []
    for c in Y:
        tmp  = []
        for yorg, yc in zip(Y_org, Y):
            if yc == c:
                tmp.append(yorg)
        subplot_titles.append("{:.2f} ~ {:.2f}".format(min(tmp),max(tmp)))
        
    fig = make_subplots(rows = 1, cols = out_dim, vertical_spacing = 0.02, subplot_titles=subplot_titles)
    for i in range(out_dim):
        fig.add_trace(f_scatter(Xc_set[i], color_set[i]) ,1,i+1)
        fig.add_trace(f_heatmap(
            z = probs[:,i].reshape(100,100), z_range=(0,1), color_bar_xy=(1.0,0.5),
        ),1,i+1)
    
    f_update_layout(fig, 1000*out_dim, 1000)
    fig.show()
    # plotly.offline.plot(fig, filename = save_img_dir + "g1.html", auto_open=False)
    

def Run(ct,*args):
    dist_thr = 0.8
    i = 2
    
    dm = read_dm("GMM12Sig8LCB4/checkpoints",i,"ch500",2)
    X_tip, Y_tip, X_shake, Y_shake, Y_tip_org, Y_shake_org = extract_trainig_data2(dm, dist_thr)
    
    model = train2(X_tip, Y_tip, in_dim = 2)
    X = torch.tensor([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ], dtype = torch.float32, requires_grad = True, device='cuda')
    preds = model(X)
    probs = torch.nn.Softmax(dim = 1)(preds)
    
    probs = probs.to('cpu').detach().numpy().copy()
    X_tip = X_tip.to('cpu').detach().numpy().copy()
    Y_tip = Y_tip.to('cpu').detach().numpy().copy()
    scatter2(X_tip, Y_tip, probs, Y_tip_org)