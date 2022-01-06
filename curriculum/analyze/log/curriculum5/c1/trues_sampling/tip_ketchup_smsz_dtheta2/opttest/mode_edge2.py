#coding: UTF-8
import torch
from torch import device, dtype, nn
from torch.cuda import max_memory_allocated
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

import dill
import matplotlib.pyplot as plt
from time import time
import numpy as np
from plotly import graph_objects as go
from sklearn import preprocessing

from .setup2 import *
from .learn3 import Domain3


NoneType = type(None)


def extract_trainig_data(dm):
    log = dm.log
    X_tip, Y_tip = [], []
    X_shake, Y_shake = [], []
    for i in log['ep']:
        x = []
        r = log['r_at_est_optparam'][i]
        c = 1 if r <-1 else 0
        if log['skill'][i] == 'tip':
            x.append(log['est_optparam'][i])
            x.append(log['smsz'][i])
            X_tip.append(x)
            Y_tip.append(c)
        else:
            x.append(log['smsz'][i])
            X_shake.append(x)
            Y_shake.append(c)
            
    
    X_tip = torch.tensor(X_tip, dtype = torch.float32)
    Y_tip = torch.tensor(Y_tip, dtype = torch.int64)
    X_shake = torch.tensor(X_shake, dtype = torch.float32)
    Y_shake = torch.tensor(Y_shake, dtype = torch.int64)
    
    return X_tip, Y_tip, X_shake, Y_shake


class BinaryClassification(nn.Module):
    def __init__(self, in_dim):
        super(BinaryClassification, self).__init__()
        n_units = 200
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, n_units),
            # nn.ReLU(),
            # nn.BatchNorm1d(n_units),
            # nn.Linear(n_units, n_units),
            # nn.ReLU(),
            # nn.BatchNorm1d(n_units),
            # nn.Linear(n_units, n_units),
            # nn.ReLU(),
            # nn.BatchNorm1d(n_units),
            # nn.Linear(n_units, n_units),
            # nn.ReLU(),
            # nn.BatchNorm1d(n_units),
            # nn.Linear(n_units, n_units),
            nn.ReLU(),
            # nn.BatchNorm1d(n_units),
            nn.Linear(n_units, 2),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
    

class Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    
def get_accuracy(y_pred, y_true):
    y_pred = y_pred.to('cpu')
    y_prob = torch.nn.Softmax(dim = 1)(y_pred)
    y_prob = torch.argmax(y_prob, axis = 1)
    y_true = y_true.to('cpu')
    return (1. * sum(y_prob == y_true) / len(y_true)).item()


def train(model, X, Y, criterion, optimizer, batch_size = 10, n_epoch = 100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    torch.backends.cudnn.benchmark = True

    dataset = Dataset(X, Y)
    dataloader = DataLoader(dataset, batch_size = batch_size)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8)
    # scheduler = ReduceLROnPlateau(optimizer, 'max')
    
    model.to(device)
    X = X.to(device)
    Y = Y.to(device)

    max_acc = 0
    for epoch in range(1, n_epoch+1):
        epoch_loss = 0
        # epoch_acc = 0
        epoch_data_len = 0
        for batch, (bX, bY) in enumerate(dataloader):
            model.train()
            bX, bY = bX.to(device), bY.to(device)

            pred = model(bX)
            loss = criterion(pred, bY)
            # acc = get_accuracy(pred, bY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss
            # epoch_acc += acc*len(bX)
            epoch_data_len += len(bX)

        model.eval()
        pred = model(X)
        epoch_acc = get_accuracy(pred, Y)
        max_acc = max(max_acc, epoch_acc)
        epoch_loss /= epoch_data_len
        # epoch_acc /= epoch_data_len
        print('Epoch {}: | Loss: {:.5f} | Acc: {:.5f} | Max Acc: {:.5f}'.format(epoch, epoch_loss, epoch_acc, max_acc))
        
        scheduler.step(epoch_loss)
        # scheduler.step(epoch_acc)
        # if epoch_acc == 1:
        #     break

    return model, max_acc


def train_test(X, Y, n_trial):
    dt_list = []
    acc_list = []
    for _ in range(n_trial):
        model = BinaryClassification()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        t1 = time()
        model, max_acc = train(model, X, Y, criterion, optimizer, batch_size = len(X), n_epoch = 1000)
        t2 = time()
        dt = t2 - t1
        dt_list.append(dt)
        acc_list.append(max_acc)
    print(np.mean(dt_list), np.std(dt_list))
    print(np.mean(acc_list), np.std(acc_list))
    
    
def compute_grad(dm, model, in_dim):
    model.eval()
    grads = []
    for i in range(2):
        if in_dim == 2:
            X = torch.tensor([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ], dtype = torch.float32, requires_grad = True, device='cuda')
        elif in_dim == 1:
            X = torch.tensor([[smsz] for smsz in dm.smsz ], dtype = torch.float32, requires_grad = True, device='cuda')
        preds = model(X)
        probs = torch.nn.Softmax(dim = 1)(preds)
        m = torch.zeros((len(X),2), device='cuda')
        m[:, i] = 1
        probs.backward(m)
        grads.append(X.grad.data.to('cpu').detach().numpy().copy())
    
    if in_dim == 2:
        sum_grad = np.abs(grads[0][:,0]) + np.abs(grads[0][:,1]) + np.abs(grads[1][:,0]) + np.abs(grads[1][:,1])
        sum_grad *= 1./4
        sum_grad = sum_grad.reshape(100,100)
    elif in_dim == 1:
        sum_grad = np.abs(grads[0]) + np.abs(grads[1])
        sum_grad *= 1./2
        sum_grad = sum_grad.reshape(100)
    
    probs = probs.to('cpu').detach().numpy().copy()
    
    return probs, grads, sum_grad


def compute_grad_mean(dm, models):
    grads = []
    for i in range(2):
        probs_concat = []
        # sum_probs = 0
        X = torch.tensor([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ], dtype = torch.float32, requires_grad = True, device='cuda')
        for model in models:
            probs_concat.append(torch.nn.Softmax(dim = 1)(model(X)))
        mean_probs = sum(probs_concat) / len(models)
        m = torch.zeros((len(X),2), device='cuda')
        m[:, i] = 1
        mean_probs.backward(m)
        grads.append(X.grad.data)
    
    grads[0] = grads[0].to('cpu').detach().numpy().copy()
    grads[1] = grads[1].to('cpu').detach().numpy().copy()
    mean_probs = mean_probs.to('cpu').detach().numpy().copy()
    sum_grad = np.abs(grads[0][:,0]) + np.abs(grads[0][:,1]) + np.abs(grads[1][:,0]) + np.abs(grads[1][:,1])
    sum_grad *= 1./4
    sum_grad = sum_grad.reshape(100,100)
    
    return mean_probs, grads, sum_grad

    
def compute_grad_test(dm, n_trial):
    model = BinaryClassification()
    model.eval()
    model.to('cuda')    
    
    dt_list = []
    for _ in range(n_trial):
        t1 = time()
        X = torch.tensor([[dtheta2, smsz] for dtheta2 in dm.dtheta2 for smsz in dm.smsz ], dtype = torch.float32, requires_grad = True, device='cuda')
        out = model(X)
        m = torch.zeros((len(X),2), device='cuda')
        m[:, 0] = 1
        out.backward(m)
        t2 = time()
        dt = t2 - t1
        dt_list.append(dt)
    print(np.mean(dt_list), np.std(dt_list))
    print(X.grad.data[-10:])
    
    
def scatter(X, Y, datotal, gmmpred, evaluation, probs, grads, sum_grad, nn_res_concat = None, mean_res = None, dm = None, save_img_dir = None):    
    X1 = np.array([x for x, y in zip(X, Y) if y == 1])
    X0 = np.array([x for x, y in zip(X, Y) if y == 0])
    if type(sum_grad) != type(None):
        loge_sum_grad = np.log(sum_grad + 1)
        log10_sum_grad = np.log10(sum_grad + 1)
        log_sum_grad = loge_sum_grad
    smsz = np.linspace(0.3,0.8,100)
    dtheta = np.linspace(0.1,1.0,100)[::-1]
    
    f_scatter = lambda X, color: go.Scatter(
        mode = 'markers', x = X[:,1], y = X[:,0],
        marker = dict(color = color, size = 12, line = dict(color = "black", width = 3)),
        showlegend = False,
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
    
    
    # グラフ1 サンプル点をプロット
    # ラベル1: red, ラベル0: black
    fig = make_subplots(rows = 4, cols = 3, horizontal_spacing = 0.02, vertical_spacing = 0.02)
    fig.add_trace(f_scatter(X0, 'black') ,1,1)
    fig.add_trace(f_scatter(X1, 'red') ,1,1)
    
    fig.add_trace(f_scatter(X0, 'black') ,1,2)
    fig.add_trace(f_scatter(X1, 'red') ,1,2)
    fig.add_trace(f_heatmap(
        z = datotal[TRUE], z_range=[0,0.55], color_bar_xy=(1.0,0.5),
        color_scale = [
            [0, "rgb(0, 0, 255)"],
            [0.2727, "rgb(0, 255, 255)"],
            [0.5454, "rgb(0, 255, 0)"],
            [0.772, "rgb(255, 255, 0)"],
            [1, "rgb(255, 0, 0)"],
        ],
    ),1,2)
    
    fig.add_trace(f_scatter(X0, 'black') ,1,3)
    fig.add_trace(f_scatter(X1, 'red') ,1,3)
    fig.add_trace(f_heatmap(
        z = datotal[NNMEAN], z_range=(0,0.55), color_bar_xy=(1.0,0.5),
        color_scale = [
            [0, "rgb(0, 0, 255)"],
            [0.2727, "rgb(0, 255, 255)"],
            [0.5454, "rgb(0, 255, 0)"],
            [0.772, "rgb(255, 255, 0)"],
            [1, "rgb(255, 0, 0)"],
        ],
    ),1,3)
    
    fig.add_trace(f_scatter(X0, 'black') ,2,1)
    fig.add_trace(f_scatter(X1, 'red') ,2,1)
    fig.add_trace(f_heatmap(
        z = datotal[NNSD], z_range=(0,0.2), color_bar_xy=(1.0,0.5),
    ),2,1)
    
    fig.add_trace(f_scatter(X0, 'black') ,2,2)
    fig.add_trace(f_scatter(X1, 'red') ,2,2)
    fig.add_trace(f_heatmap(
        z = gmmpred, z_range=(0,0.2), color_bar_xy=(1.0,0.5),
    ),2,2)
    
    fig.add_trace(f_scatter(X0, 'black') ,2,3)
    fig.add_trace(f_scatter(X1, 'red') ,2,3)
    fig.add_trace(f_heatmap(
        z = evaluation, z_range=(-5,0), color_bar_xy=(1.0,0.5),
    ),2,3)
    
    fig.add_trace(f_scatter(X0, 'black') ,3,1)
    fig.add_trace(f_scatter(X1, 'red') ,3,1)
    fig.add_trace(f_heatmap(
        z = dm.datotal[TIP][RFUNC], z_range=(-5,0), color_bar_xy=(1.0,0.5),
    ),3,1)
    
    fig.add_trace(f_scatter(X0, 'black') ,3,2)
    fig.add_trace(f_scatter(X1, 'red') ,3,2)
    fig.add_trace(f_heatmap(
        z = probs[:,1].reshape(100,100), z_range=(0,1), color_bar_xy=(1.0,0.5),
    ),3,2)
    
    fig.add_trace(f_scatter(X0, 'black') ,3,3)
    fig.add_trace(f_scatter(X1, 'red') ,3,3)
    fig.add_trace(f_heatmap(
        z = log10_sum_grad, z_range=(0,2), color_bar_xy=(1.0,0.5),
    ),3,3)
    
    # fig.add_trace(f_scatter(X0, 'black') ,4,3)
    # fig.add_trace(f_scatter(X1, 'red') ,4,3)
    # fig.add_trace(f_heatmap(
    #     z = evaluation - log10_sum_grad, z_range=(-5,0), color_bar_xy=(1.0,0.5),
    # ),4,3)
    fig.add_trace(f_scatter(X0, 'black') ,4,3)
    fig.add_trace(f_scatter(X1, 'red') ,4,3)
    fig.add_trace(f_heatmap(
        z = evaluation - loge_sum_grad, z_range=(-5,0), color_bar_xy=(1.0,0.5),
    ),4,3)
    
    f_update_layout(fig, 2400, 3200)
    # fig.show()
    plotly.offline.plot(fig, filename = save_img_dir + "g1.html", auto_open=False)
    
    
    # グラフ2 preds のヒートマップとサンプル点
    fig = make_subplots(rows = 2, cols = 3, horizontal_spacing = 0.045, vertical_spacing = 0.05)
    fig.add_trace(f_scatter(X0, 'black') ,1,1)
    fig.add_trace(f_scatter(X1, 'red') ,1,1)
    fig.add_trace(f_heatmap(z = probs[:,0].reshape(100,100), z_range=(0.,1), color_bar_xy=(0.3,0.75), color_scale=None),1,1)
    fig.add_trace(f_scatter(X0, 'black') ,1,2)
    fig.add_trace(f_scatter(X1, 'red') ,1,2)
    fig.add_trace(f_heatmap(z = grads[0][:,1].reshape(100,100), z_range=(-10,10), color_bar_xy=(0.67,0.75), color_scale=None),1,2)
    fig.add_trace(f_scatter(X0, 'black') ,1,3)
    fig.add_trace(f_scatter(X1, 'red') ,1,3)
    fig.add_trace(f_heatmap(z = grads[0][:,0].reshape(100,100), z_range=(-10,10), color_bar_xy=(1.,0.75), color_scale=None),1,3)
    
    fig.add_trace(f_scatter(X0, 'black') ,2,1)
    fig.add_trace(f_scatter(X1, 'red') ,2,1)
    fig.add_trace(f_heatmap(z = probs[:,1].reshape(100,100), z_range=(0,1), color_bar_xy=(0.3,0.25), color_scale=None),2,1)
    fig.add_trace(f_scatter(X0, 'black') ,2,2)
    fig.add_trace(f_scatter(X1, 'red') ,2,2)
    fig.add_trace(f_heatmap(z = grads[1][:,1].reshape(100,100), z_range=(-10,10), color_bar_xy=(0.67,0.25), color_scale=None),2,2)
    fig.add_trace(f_scatter(X0, 'black') ,2,3)
    fig.add_trace(f_scatter(X1, 'red') ,2,3)
    fig.add_trace(f_heatmap(z = grads[1][:,0].reshape(100,100), z_range=(-10,10), color_bar_xy=(1.,0.25), color_scale=None),2,3)
    f_update_layout(fig, 3000, 1600)
    # fig.show()
    plotly.offline.plot(fig, filename = save_img_dir + "g2.html", auto_open=False)
    
    
    p = probs[:,0].reshape(100,100)
    gx = grads[0][:,1].reshape(100,100)
    v_list = [0.8,0.6,0.4,0.2]
    fig = make_subplots(rows = len(v_list), cols = 2, horizontal_spacing = 0.045, vertical_spacing = 0.05)
    for i, v in enumerate(v_list):
        fig.add_trace(go.Scatter(x = smsz, y = p[idx_of_the_nearest(dtheta,v),:], mode = 'lines', line = dict(color = 'black', width = 3)),i+1,1)
        fig.add_trace(go.Scatter(x = smsz, y = gx[idx_of_the_nearest(dtheta,v),:], mode = 'lines', line = dict(color = 'black', width = 3)),i+1,2)
    f_update_layout(fig, 1600, 600*len(v_list))
    plotly.offline.plot(fig, filename = save_img_dir + "g2_debug.html", auto_open=False)
    
    
    # グラフ3 勾配
    fig = make_subplots(rows = 2, cols = 4, horizontal_spacing = 0.045, vertical_spacing = 0.05)
    fig.add_trace(f_scatter(X0, 'black') ,1,1)
    fig.add_trace(f_scatter(X1, 'red') ,1,1)
    fig.add_trace(f_heatmap(z = probs[:,1].reshape(100,100), z_range=(0,1), color_bar_xy=(0.25,0.75), color_scale=None),1,1)
    fig.add_trace(f_scatter(X0, 'black') ,1,2)
    fig.add_trace(f_scatter(X1, 'red') ,1,2)
    fig.add_trace(f_heatmap(z = sum_grad, z_range=(0,40), color_bar_xy=(0.48,0.75), color_scale=None),1,2)
    fig.add_trace(f_scatter(X0, 'black') ,1,3)
    fig.add_trace(f_scatter(X1, 'red') ,1,3)
    fig.add_trace(f_heatmap(z = loge_sum_grad, z_range=(0,4), color_bar_xy=(0.74,0.75), color_scale=None),1,3)
    fig.add_trace(f_scatter(X0, 'black') ,1,4)
    fig.add_trace(f_scatter(X1, 'red') ,1,4)
    fig.add_trace(f_heatmap(z = log10_sum_grad, z_range=(0,2), color_bar_xy=(1.0,0.75), color_scale=None),1,4)
    f_update_layout(fig, 4000, 1400)    
    
    outputs = log_sum_grad
    out1 = outputs[:,idx_of_the_nearest(smsz, 0.3)]
    out2 = outputs[:,idx_of_the_nearest(smsz, 0.45)]
    out3 = outputs[:,idx_of_the_nearest(smsz, 0.6)]
    fig.add_trace(go.Scatter(x = dtheta, y = out1, mode = 'lines', line = dict(color = 'black', width = 3)),2,1)
    fig.add_trace(go.Scatter(x = dtheta, y = out2, mode = 'lines', line = dict(color = 'black', width = 3)),2,2)
    fig.add_trace(go.Scatter(x = dtheta, y = out3, mode = 'lines', line = dict(color = 'black', width = 3)),2,3)
    # fig.show()
    plotly.offline.plot(fig, filename = save_img_dir + "g3.html", auto_open=False)
    
    
    # # グラフ4 学習のばらつき
    # n_samples = 3
    # fig = make_subplots(rows = 2, cols = n_samples, horizontal_spacing = 0.045, vertical_spacing = 0.05)
    # for i, (probs, grads, sum_grad) in enumerate(nn_res_concat[:n_samples]):
    #     log10_sum_grad = np.log10(sum_grad + 1)
        
    #     fig.add_trace(f_scatter(X0, 'black') ,1,i+1)
    #     fig.add_trace(f_scatter(X1, 'red') ,1,i+1)
    #     fig.add_trace(f_heatmap(z = probs[:,1].reshape(100,100), z_range=(0,1), color_bar_xy=(0.25,0.75), color_scale=None),1,i+1)
        
    #     fig.add_trace(f_scatter(X0, 'black') ,2,i+1)
    #     fig.add_trace(f_scatter(X1, 'red') ,2,i+1)
    #     fig.add_trace(f_heatmap(z = log10_sum_grad, z_range=(0,2), color_bar_xy=(0.25,0.25), color_scale=None),2,i+1)
    # f_update_layout(fig, 800*(i+1), 1600)
    # fig.show()
    # plotly.offline.plot(fig, filename = save_img_dir + "g4.html", auto_open=False)
    
    
    # # グラフ5 モデルの平均化
    # log10_sum_grad = np.mean([np.log10(sum_grad + 1) for _,_,sum_grad in nn_res_concat], axis = 0)
    # prob = np.mean([probs[:,1].reshape(100,100) for probs,_,_ in nn_res_concat], axis = 0)
    
    # fig = make_subplots(rows = 1, cols = 2, horizontal_spacing = 0.045, vertical_spacing = 0.05)
    # fig.add_trace(f_scatter(X0, 'black') ,1,1)
    # fig.add_trace(f_scatter(X1, 'red') ,1,1)
    # fig.add_trace(f_heatmap(z = prob, z_range=(0,1), color_bar_xy=(0.25,0.75), color_scale=None),1,1)
    
    # fig.add_trace(f_scatter(X0, 'black') ,1,2)
    # fig.add_trace(f_scatter(X1, 'red') ,1,2)
    # fig.add_trace(f_heatmap(z = log10_sum_grad, z_range=(0,1), color_bar_xy=(0.25,0.75), color_scale=None),1,2)
    
    # f_update_layout(fig, 1600, 800)
    # fig.show()
    # plotly.offline.plot(fig, filename = save_img_dir + "g5.html", auto_open=False)
    
    
    # # グラフ6 モデルの平均化 修正版
    # mean_prob = mean_res[0][:,1].reshape(100,100)
    # mean_sum_grad = mean_res[1]
    # log10_mean_sum_grad = np.log10(mean_sum_grad + 1)
    
    # fig = make_subplots(rows = 1, cols = 2, horizontal_spacing = 0.045, vertical_spacing = 0.05)
    # fig.add_trace(f_scatter(X0, 'black') ,1,1)
    # fig.add_trace(f_scatter(X1, 'red') ,1,1)
    # fig.add_trace(f_heatmap(z = mean_prob, z_range=(0,1), color_bar_xy=(0.25,0.75), color_scale=None),1,1)
    
    # fig.add_trace(f_scatter(X0, 'black') ,1,2)
    # fig.add_trace(f_scatter(X1, 'red') ,1,2)
    # fig.add_trace(f_heatmap(z = log10_mean_sum_grad, z_range=(0,2), color_bar_xy=(0.25,0.75), color_scale=None),1,2)
    
    # f_update_layout(fig, 1600, 800)
    # fig.show()
    # plotly.offline.plot(fig, filename = save_img_dir + "g6.html", auto_open=False)
    
    
    # グラフ7 3Dグラフ
    fig = make_subplots(rows = 1, cols = 2, horizontal_spacing = 0.045, vertical_spacing = 0.05, specs = [[{"type": "surface"}, {"type": "surface"}]])
    f_update_layout(fig, 2000, 1000)
    fig.add_trace(go.Surface(
        z = probs[:,0].reshape(100,100), x = smsz, y = dtheta,
        cmin = 0., cmax = 1, 
        colorbar = dict(len = 0.4),
        showlegend = False,
    ),1,1)
    fig.add_trace(go.Surface(
        z = probs[:,0].reshape(100,100), x = smsz, y = dtheta,
        cmin = 0., cmax = 1, 
        colorbar = dict(len = 0.4),
        showlegend = False,
    ),1,2)
    fig['layout']['scene2']['xaxis_autorange'] = 'reversed'
    fig['layout']['scene2']['yaxis_autorange'] = 'reversed'
    fig['layout']['scene2']['zaxis_autorange'] = 'reversed'
    fig.update_layout(scene1 = dict(aspectratio={"x":1,"y":1,"z":1}), scene2 = dict(aspectratio={"x":1,"y":1,"z":1}))
    plotly.offline.plot(fig, filename = save_img_dir + "g7.html", auto_open=False)
    
    
def scatter_shake(X, Y, datotal, gmmpred, evaluation, probs, grads, sum_grad, nn_res_concat = None, mean_res = None, save_img_dir = None):
    XY1 = np.array([[x.item(), y.item()] for x, y in zip(X, Y) if y == 1])
    XY0 = np.array([[x.item(), y.item()] for x, y in zip(X, Y) if y == 0])
    log10_sum_grad = np.log10(sum_grad + 1)
    loge_sum_grad = np.log(sum_grad + 1)
    log_sum_grad = loge_sum_grad
    smsz = np.linspace(0.3,0.8,100)
    
    f_scatter = lambda XY, color: go.Scatter(
        mode = 'markers', x = XY[:,0] if not len(XY) == 0 else [], y = XY[:,1] if not len(XY) == 0 else [],
        marker = dict(color = color, size = 12, line = dict(color = "black", width = 3)),
        showlegend = False,
    )
    
        
    # グラフ1
    fig = make_subplots(rows = 2, cols = 1, horizontal_spacing = 0.045, vertical_spacing = 0.05)
    
    fig.add_trace(f_scatter(XY0, 'black'), 1,1)
    fig.add_trace(f_scatter(XY1, 'red'), 1,1)
    fig.add_trace(go.Scatter(x = smsz, y = probs[:,1], mode='lines', line=dict(color='orange')), 1,1)
    fig.add_trace(go.Scatter(x = smsz, y = log_sum_grad), 2,1)
    
    # fig.show()
    plotly.offline.plot(fig, filename = save_img_dir + "g_shake.html", auto_open=False)
    
    
def eval_save(evaluation, save_dir, name):
    save_path = save_dir + name
    with open(save_dir + name +'.pickle', mode = 'wb') as f:
        pickle.dump(evaluation, f)
        

def eval_load(save_dir, name):
    save_path = save_dir + name
    with open(save_dir + name +'.pickle', mode = 'rb') as f:
        evaluation = dill.load(f)
        
    return evaluation


def grad_penalty(sum_grad):
    return np.log(sum_grad + 1)

    
def eval_tip(datotal, dm, mode = 'Er', gmmpred = None, sum_grad = None, logdir = None, update = False, gmm_eval = None):
    if logdir and os.path.exists(logdir + mode + '_shake.pickle') and not update and type(gmm_eval) == type(None):
        return eval_load(logdir, mode + '_tip')
    
    datotal_nnmean = datotal[TIP][NNMEAN]
    datotal_nnsd = datotal[TIP][NNSD]
    
    sd_gain = 1
    if mode == 'Er':
        gmm = np.zeros((100,100))
    elif mode == 'LCB':
        gmm = np.zeros((100,100))
        LCB_ratio = dm.LCB_ratio
    elif mode == 'GMM':
        gmm = gmmpred[TIP]
        LCB_ratio = dm.LCB_ratio
    elif mode == 'Edge':
        gmm = np.zeros((100,100))
        LCB_ratio = dm.LCB_ratio
        edge_ratio = 1.0
        # grad_penalty = np.log10(sum_grad + 1)
    elif mode == 'Edge_GMM':
        edge_ratio = 1.0
        # edge_ratio = 3.0
        # grad_penalty = np.log10(sum_grad + 1)
        if type(gmm_eval) != type(None):
            return gmm_eval - edge_ratio*grad_penalty(sum_grad)
        else:
            gmm = gmmpred[TIP]
            LCB_ratio = dm.LCB_ratio
    elif mode == 'Edge_LCB':
        edge_ratio = 1.0
        # grad_penalty = np.log10(sum_grad + 1)
        eval_lcb = eval_tip(datotal, dm, mode='LCB', logdir=logdir)
        return eval_lcb - edge_ratio*grad_penalty(sum_grad)
    
    rmodel = Rmodel("Fdatotal_gentle")
    rnn_sm = np.array([[rmodel.Predict(x=[0.3, datotal_nnmean[idx_dtheta2, idx_smsz]], x_var=[0, (sd_gain*(datotal_nnsd[idx_dtheta2, idx_smsz] + gmm[idx_dtheta2, idx_smsz]))**2], with_var=True) for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
    Er = np.array([[rnn_sm[idx_dtheta2, idx_smsz].Y.item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
    Sr = np.sqrt([[rnn_sm[idx_dtheta2, idx_smsz].Var[0,0].item() for idx_smsz in range(100)] for idx_dtheta2 in range(100)])
    
    if mode == 'Er':
        evaluation = Er
    elif mode == 'LCB' or mode == 'GMM':
        evaluation = Er - LCB_ratio*Sr
    elif mode == 'Edge' or mode == 'Edge_GMM':
        evaluation = Er - LCB_ratio*Sr - edge_ratio*grad_penalty(sum_grad)
    
    if logdir:
        eval_save(evaluation, logdir, mode + '_tip')
        
    return evaluation


def eval_shake(datotal, dm, mode = 'Er', gmmpred = None, sum_grad = None, logdir = None, update = False, gmm_eval = None):
    if logdir and os.path.exists(logdir + mode + '_shake.pickle') and not update and type(gmm_eval) == type(None):
        return eval_load(logdir, mode + '_shake')
    
    datotal_nnmean = datotal[SHAKE][NNMEAN]
    datotal_nnsd = datotal[SHAKE][NNSD]
    
    sd_gain = 1
    if mode == 'Er':
        gmm = np.zeros(100)
    elif mode == 'LCB':
        gmm = np.zeros(100)
        LCB_ratio = dm.LCB_ratio
    elif mode == 'GMM':
        gmm = gmmpred[SHAKE]
        LCB_ratio = dm.LCB_ratio
    elif mode == 'Edge_GMM':
        edge_ratio = 1.0
        # grad_penalty = np.log10(sum_grad + 1)
        if type(gmm_eval) != type(None):
            return gmm_eval - edge_ratio*grad_penalty(sum_grad)
        else:
            gmm = gmmpred[SHAKE]
            LCB_ratio = dm.LCB_ratio
    elif mode == 'Edge_LCB':
        edge_ratio = 1.0
        # grad_penalty = np.log10(sum_grad + 1)
        # grad_penalty = np.log(sum_grad + 1)
        eval_lcb = eval_shake(datotal, dm, mode='LCB', logdir=logdir)
        return eval_lcb - edge_ratio*grad_penalty(sum_grad)
    
    rmodel = Rmodel("Fdatotal_gentle")
    rnn_sm = np.array([rmodel.Predict(x=[0.3, datotal_nnmean[idx_smsz]], x_var=[0, (sd_gain*(datotal_nnsd[idx_smsz] + gmm[idx_smsz]))**2], with_var=True) for idx_smsz in range(100)])
    Er = np.array([rnn_sm[idx_smsz].Y.item() for idx_smsz in range(100)])
    Sr = np.sqrt([rnn_sm[idx_smsz].Var[0,0].item() for idx_smsz in range(100)])
    
    if mode == 'Er':
        evaluation = Er
    elif mode == 'LCB' or mode == 'GMM':
        evaluation = Er - LCB_ratio*Sr
    elif mode == 'Edge_GMM':
        evaluation = Er - LCB_ratio*Sr - edge_ratio*grad_penalty(sum_grad)
    
    if logdir:
        eval_save(evaluation, logdir, mode + '_shake')
        
    return evaluation
    
    
def optimize_check(dm, datotal, evaluation, logdir, save_img_dir):
    tip_er = eval_tip(datotal, dm, mode='Er', logdir=logdir)
    tip_lcb = eval_tip(datotal, dm, mode='LCB', logdir=logdir)
    # tip_gmm = eval_tip(datotal, dm, mode='GMM', logdir=logdir)
    tip_gmm = evaluation[TIP]
    # tip_edge = eval_tip(datotal, dm, mode='Edge', logdir=logdir)
    # tip_edge_gmm = eval_tip(datotal, dm, mode='Edge_GMM', logdir=logdir)
    with open(logdir + 'nn_res_tip.pickle', mode = 'rb') as f:
        probs, grads, sum_grad = dill.load(f)
    tip_edge_gmm = eval_tip(datotal, dm=dm, mode='Edge_GMM', sum_grad=sum_grad, gmm_eval=tip_gmm)
    tip_edge_lcb = eval_tip(datotal, dm=dm, mode='Edge_LCB', sum_grad=sum_grad, gmm_eval=tip_gmm)
    
    shake_er = eval_shake(datotal, dm, mode='Er', logdir=logdir)
    shake_lcb = eval_shake(datotal, dm, mode='LCB', logdir=logdir)
    # shake_gmm = eval_shake(datotal, dm, mode='GMM', logdir=logdir)
    shake_gmm = evaluation[SHAKE]
    # shake_edge_gmm = eval_shake(datotal, dm, mode='Edge_GMM', logdir=logdir)
    # shake_edge_gmm = eval_shake(datotal, dm, mode='Edge_GMM', logdir=logdir, gmm_eval=shake_gmm)
    with open(logdir + 'nn_res_shake.pickle', mode = 'rb') as f:
        probs, grads, sum_grad = dill.load(f)
    shake_edge_gmm = eval_shake(datotal, dm=dm, mode='Edge_GMM', sum_grad=sum_grad, gmm_eval=shake_gmm)
    shake_edge_lcb = eval_shake(datotal, dm=dm, mode='Edge_LCB', sum_grad=sum_grad, gmm_eval=shake_gmm)
    
    def evaluation(tip, shake):
        opt_tip = np.max(tip, axis = 0)
        opt_shake = shake
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(tip, axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        optr = [true_ytip[i] if yt > ys else true_yshake[i] for i, (yt, ys) in enumerate(zip(opt_tip, opt_shake))]
        color = ["red" if yt > ys else "purple" for yt, ys in zip(opt_tip, opt_shake)]
        return opt_tip, opt_shake, optr, color
    
    fig = make_subplots(rows = 3, cols = 3, horizontal_spacing = 0.05, vertical_spacing = 0.05)
    for i, (tip, shake) in enumerate(zip([tip_lcb, tip_edge_lcb, tip_gmm], [shake_lcb, shake_edge_lcb, shake_gmm])):
        opt_tip, opt_shake, optr, color = evaluation(tip, shake)
        opt_smsz = []
        opt_dtheta = []
        for idx_smsz in range(100):
            if opt_tip[idx_smsz] >= opt_shake[idx_smsz]:
                opt_smsz.append(dm.smsz[idx_smsz])
                opt_dtheta.append(dm.dtheta2[np.argmax(tip[:,idx_smsz])])
        
        fig.add_trace(go.Scatter(x = dm.smsz, y = optr, mode = "markers", showlegend = False, marker = dict(size = 16, color = color), ), i+1,1)
        fig.add_trace(go.Scatter(x = dm.smsz, y = opt_tip, mode = "lines", line = dict(color = "red", width = 4), ), i+1,1)
        fig.add_trace(go.Scatter(x = dm.smsz, y = opt_shake, mode = "lines", line = dict(color = "purple", width = 4), ), i+1,1)
        
        fig.add_trace(go.Heatmap(z = tip, x = dm.smsz, y = dm.dtheta2, zmin = -5, zmax = 0, colorbar=dict(len = 0.2), colorscale="Viridis"), i+1,2)
        fig.add_trace(go.Scatter(x = opt_smsz, y = opt_dtheta, mode = "markers", marker = dict(color = "red", size = 16, line = dict(width = 2)),), i+1,2)
        
        fig.add_trace(go.Heatmap(z = dm.datotal[TIP][RFUNC], x = dm.smsz, y = dm.dtheta2, zmin = -5, zmax = 0, colorbar=dict(len = 0.2), colorscale="Viridis"), i+1,3)
        fig.add_trace(go.Scatter(x = opt_smsz, y = opt_dtheta, mode = "markers", marker = dict(color = "red", size = 16, line = dict(width = 2)), ), i+1,3)
        
        fig['layout']['xaxis'+str(3*i+1)].update(dict(linecolor = "black", range= (0.29,0.82), title = 'size'))
        fig['layout']['yaxis'+str(3*i+1)].update(dict(linecolor = "black", range= (-4,0.2), title = 'size'))
    fig.update_layout(
        plot_bgcolor = "white", font = dict(size = 20),
        showlegend = False, width = 2400, height = 2400,
    )
    # fig.show()
    
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "optimize_check.html", auto_open=False)
    
    
def opttest_comp_use_edge(name, n, ch = None, ver = 2, use_edge = True, without_gmm = True, suffix = ""):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/opttest_comp/{}/edge_test/".format(ver, name, ch)
    check_or_create_dir(save_img_dir)
    
    y_concat = []
    yest_concat = {TIP: [], SHAKE: []}
    for i in range(1,n):
    # for i in range(1,25)+range(26,80)+range(90,96)+range(98,100):
        logdir = basedir + "{}/t{}/{}".format(name, i, ch)
        print(logdir)
        dm = Domain3.load(logdir+"dm.pickle")
        datotal = setup_datotal(dm, logdir)
        if use_edge:
            evaluation = dict()
            if not without_gmm:
                with open(logdir + 'nn_res_tip.pickle', mode = 'rb') as f:
                    probs, grads, sum_grad = dill.load(f)
                evaluation[TIP] = eval_tip(datotal, dm=dm, mode='Edge_GMM', sum_grad=sum_grad, logdir=logdir, gmm_eval=setup_eval(dm, logdir)[TIP])
                with open(logdir + 'nn_res_shake.pickle', mode = 'rb') as f:
                    probs, grads, sum_grad = dill.load(f)
                # evaluation[SHAKE] = eval_shake(datotal, dm, mode='GMM', logdir=logdir)
                evaluation[SHAKE] = eval_shake(datotal, dm=dm, mode='Edge_GMM', sum_grad=sum_grad, logdir=logdir, gmm_eval=setup_eval(dm, logdir)[SHAKE])
            else:
                with open(logdir + 'nn_res_tip.pickle', mode = 'rb') as f:
                    probs, grads, sum_grad = dill.load(f)
                evaluation[TIP] = eval_tip(datotal, dm=dm, mode='Edge_LCB', sum_grad=sum_grad, logdir=logdir, gmm_eval=setup_eval(dm, logdir)[TIP])
                with open(logdir + 'nn_res_shake.pickle', mode = 'rb') as f:
                    probs, grads, sum_grad = dill.load(f)
                # evaluation[SHAKE] = eval_shake(datotal, dm, mode='GMM', logdir=logdir)
                evaluation[SHAKE] = eval_shake(datotal, dm=dm, mode='Edge_LCB', sum_grad=sum_grad, logdir=logdir, gmm_eval=setup_eval(dm, logdir)[SHAKE])
        else:
            evaluation = setup_eval(dm, logdir)
            # evaluation = dict()
            # evaluation[TIP] = eval_tip(datotal, dm, mode='GMM', logdir=logdir)
            # evaluation[SHAKE] = eval_shake(datotal, dm, mode='GMM', logdir=logdir)
        
        true_ytip = [smsz_r[opt_idx] for i, (smsz_r, opt_idx) in enumerate(zip(dm.datotal[TIP][RFUNC].T, np.argmax(evaluation[TIP], axis = 0)))]
        true_yshake = dm.datotal[SHAKE][RFUNC]
        est_ytip = np.max(evaluation[TIP], axis=0)
        est_yshake = evaluation[SHAKE]
        
        y = []
        yest = {TIP: [], SHAKE: []}
        for idx_smsz in range(len(dm.smsz)):
            if est_ytip[idx_smsz] > est_yshake[idx_smsz]:
                y.append(true_ytip[idx_smsz])
            else:
                y.append(true_yshake[idx_smsz])
            yest[TIP].append(est_ytip[idx_smsz])
            yest[SHAKE].append(est_yshake[idx_smsz])
        y_concat.append(y)
        for skill in [TIP, SHAKE]:
            yest_concat[skill].append(yest[skill])
    ymean = np.mean(y_concat, axis = 0)
    ysd = np.std(y_concat, axis = 0)
    yestmean = dict()
    yestsd = dict()
    yp = dict()
    yestp = defaultdict(lambda: dict())
    for skill in [TIP, SHAKE]:
        yestmean[skill] = np.mean(yest_concat[skill], axis = 0)
        yestsd[skill] = np.std(yest_concat[skill], axis = 0)
    for p in [0,2,5,10,50,90,95,98,100,25,75]:
        yp[p] = np.percentile(y_concat, p, axis = 0)
        
        # if p in [5,10,50,90,95]:
        #     Print('{}percentile:'.format(p))
        #     for i in range(5):    
        #         Print('smsz_idx {}~{}:'.format(20*i,20*(i+1)), np.mean(yp[p][20*i:20*(i+1)]))
        
        for skill in[TIP, SHAKE]:
            yestp[skill][p] = np.percentile(np.array(yest_concat[skill]), p, axis = 0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = np.max(dm.datotal[TIP][RFUNC], axis = 0),
        mode = "lines",
        name = "reward (TIP)",
        line = dict(width = 2, color = "blue"),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = dm.datotal[SHAKE][RFUNC],
        mode = "lines",
        name = "reward (SHAKE)",
        line = dict(width = 2, color = "red"),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yp[50],
        mode = "markers",
        name = "reward at opt param (5%, 50%, 95%)",
        marker = dict(color = 'black'),
        error_y=dict(
            type="data",
            symmetric=False,
            array=yp[95]-yp[50],
            arrayminus=yp[50]-yp[5],
            thickness=1.5,
            width=3,
            color = 'black',
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = ymean,
        mode = "lines",
        name = "reward at opt param (mean)",
        line = dict(width = 3, color = "black"),
    ))
    badr = [len([yi for yi in y if yi < true_yshake[idx_smsz]]) for idx_smsz, y in enumerate(np.array(y_concat).T)]
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = np.ones(len(dm.smsz))*0.1+np.array([0.1 if i%2==0 else 0 for i in range(len(dm.smsz))]),
        mode = "lines+text",
        text = ["{:.0f}".format(1.*b/len(y_concat)*100) if b != 0 else "" for b in badr],
        line = dict(width=0),
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = ymean,
        mode = "markers",
        name = "reward at opt param",
        error_y=dict(
            type="data",
            symmetric=False,
            array=np.zeros(len(ysd)),
            arrayminus=ysd,
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestmean[TIP],
        mode = "markers",
        name = "evaluation (TIP) at opt param",
        error_y=dict(
            type="data",
            symmetric=True,
            array=yestsd[TIP],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestmean[SHAKE],
        mode = "markers",
        name = "evaluation (SHAKE)",
        error_y=dict(
            type="data",
            symmetric=True,
            array=yestsd[SHAKE],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[TIP][50],
        mode = "markers",
        name = "evaluation (TIP) at opt param (0%, 100%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[TIP][100]-yestp[TIP][50],
            arrayminus=yestp[TIP][50]-yestp[TIP][0],
            thickness=0.8,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[TIP][50],
        mode = "markers",
        name = "evaluation (TIP) at opt param (5%, 50%, 95%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[TIP][95]-yestp[TIP][50],
            arrayminus=yestp[TIP][50]-yestp[TIP][5],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[SHAKE][50],
        mode = "markers",
        name = "evaluation (SHAKE) at opt param (0%, 100%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[SHAKE][100]-yestp[SHAKE][50],
            arrayminus=yestp[SHAKE][50]-yestp[SHAKE][0],
            thickness=0.8,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig.add_trace(go.Scatter(
        x = dm.smsz, y = yestp[SHAKE][50],
        mode = "markers",
        name = "evaluation (SHAKE) at opt param (5%, 50%, 95%)",
        error_y=dict(
            type="data",
            symmetric=False,
            array=yestp[SHAKE][95]-yestp[SHAKE][50],
            arrayminus=yestp[SHAKE][50]-yestp[SHAKE][5],
            thickness=1.5,
            width=3,
        ),
        text = ["<br />".join(["t{}: {}".format(i+1, yi) for i, yi in enumerate(y) if yi < -1.0]) for y in np.array(y_concat).T],
    ))
    fig['layout']['yaxis']['range'] = (-5,0.5)
    fig['layout']['xaxis']['title'] = "size_srcmouth"
    fig['layout']['yaxis']['title'] = "Evaluation / Reward"
    fig['layout']['font']['size'] = 18
    
    # fig.show()
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "edge{}_gmm{}_opttest{}.html".format(str(use_edge), str(not without_gmm), suffix), auto_open=False)
    
    
    fig = go.Figure()
    fig.update_layout(
        height = 600,
    )
    for i in range(5):
        s_idx, e_idx = 20*i, 20*(i+1)
        fig.add_trace(go.Box(
                # y0 = np.array(dm.smsz)[j] 
                # y0 = n_i,
                # y0 = vis_names[::-1][n_i],
                y0 = '{:.1f}~{:.1f}'.format(dm.smsz[s_idx], dm.smsz[e_idx-1]),
                upperfence = [np.mean(yp[95][s_idx:e_idx])],
                q3 = [np.mean(yp[90][s_idx:e_idx])],
                median = [np.mean(yp[50][s_idx:e_idx])],
                q1 = [np.mean(yp[10][s_idx:e_idx])],
                lowerfence = [np.mean(yp[5][s_idx:e_idx])],            
                # fillcolor = "white",
                marker = dict(color = 'skyblue'),
                # marker = dict(color = colors[::-1][n_i], line = dict(outliercolor = colors[::-1][n_i])),
                line = dict(width = 4),
                width = 0.4,
                # whiskerwidth = 1,
                showlegend = False,
        ))
        fig.add_trace(go.Box(
                # y0 = np.array(dm.smsz)[j] 
                # y0 = n_i,
                # y0 = vis_names[::-1][n_i],
                y0 = '{:.1f}~{:.1f}'.format(dm.smsz[s_idx], dm.smsz[e_idx-1]),
                upperfence = [np.mean(yp[95][s_idx:e_idx])],
                q3 = [np.mean(yp[75][s_idx:e_idx])],
                median = [np.mean(yp[50][s_idx:e_idx])],
                q1 = [np.mean(yp[25][s_idx:e_idx])],
                lowerfence = [np.mean(yp[5][s_idx:e_idx])],            
                fillcolor = "white",
                marker = dict(color = 'blue'),
                # marker = dict(color = colors[::-1][n_i], line = dict(outliercolor = colors[::-1][n_i])),
                line = dict(width = 12),
                width = 0.8,
                # whiskerwidth = 1,
                showlegend = False,
        ))
    fig['layout']['xaxis']['range'] = (-2.8,0)
    fig['layout']['xaxis']['title'] = "reward"
    fig['layout']['xaxis']['color'] = "black"
    fig['layout']['yaxis']['color'] = "black"
    fig['layout']['font']['size'] = 42
    fig.update_layout(
        plot_bgcolor = "white",
        xaxis = dict(linecolor = "black"),
        yaxis = dict(linecolor = "black"),
    )
    
    # fig.show()
    check_or_create_dir(save_img_dir)
    plotly.offline.plot(fig, filename = save_img_dir + "edge{}_gmm{}_bar{}.html".format(str(use_edge), str(not without_gmm), suffix), auto_open=False)
    


def train_check(name, i, ch, ver = 2):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    logdir = '{}/t{}/{}/'.format(name, i, ch)
    logdir = basedir + logdir
    dm_path = logdir + 'dm.pickle'
    with open(dm_path, mode="rb") as f:
        dm = dill.load(f)
    datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
    
    X_tip, Y_tip, X_shake, Y_shake = extract_trainig_data(dm)
    
    model_tip = BinaryClassification(2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_tip.parameters(), lr=0.01)
    model_tip, max_acc = train(model_tip, X_tip, Y_tip, criterion, optimizer, batch_size = len(X_tip), n_epoch = 200)
    probs, grads, sum_grad = compute_grad(dm, model_tip, 2)
    
    with open(logdir + 'nn_res_tip.pickle', mode = 'wb') as f:
        pickle.dump([probs, grads, sum_grad], f)
    with open(logdir + 'nn_res_tip.pickle', mode = 'rb') as f:
        probs, grads, sum_grad = dill.load(f)
        
    # eval_tip(datotal, dm, mode='Er', gmmpred=None, sum_grad=None, logdir=logdir, update = True)
    # eval_tip(datotal, dm, mode='Edge_GMM', gmmpred=gmmpred, sum_grad=sum_grad, logdir=logdir, update = True)
    
    model_shake = BinaryClassification(1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_shake.parameters(), lr=0.01)
    model_shake, max_acc = train(model_shake, X_shake, Y_shake, criterion, optimizer, batch_size = len(X_shake), n_epoch = 200)
    probs, grads, sum_grad = compute_grad(dm, model_shake, 1)
    
    with open(logdir + 'nn_res_shake.pickle', mode = 'wb') as f:
        pickle.dump([probs, grads, sum_grad], f)
    with open(logdir + 'nn_res_shake.pickle', mode = 'rb') as f:
        probs, grads, sum_grad = dill.load(f)
        
    # eval_shake(datotal, dm, mode='Er', gmmpred=None, sum_grad=None, logdir=logdir, update = True)
    # eval_shake(datotal, dm, mode='Edge_GMM', gmmpred=gmmpred, sum_grad=sum_grad, logdir=logdir, update = True)
    
    
def check(name, i, ch = None, ver = 2):
    save_img_dir = PICTURE_DIR + "opttest/onpolicy{}/{}/opttest_comp/{}/t{}/".format(ver, name, ch, i)
    check_or_create_dir(save_img_dir)
    
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy{}/".format(ver)
    logdir = '{}/t{}/{}/'.format(name, i, ch)
    logdir = basedir + logdir
    dm_path = logdir + 'dm.pickle'
    with open(dm_path, mode="rb") as f:
        dm = dill.load(f)
    datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
        
    X_tip, Y_tip, X_shake, Y_shake = extract_trainig_data(dm)
    X_tip = X_tip.to('cpu').detach().numpy().copy()
    Y_tip = Y_tip.to('cpu').detach().numpy().copy()
    X_shake = X_shake.to('cpu').detach().numpy().copy()
    Y_shake = Y_shake.to('cpu').detach().numpy().copy()
    
    
    with open(logdir + 'nn_res_tip.pickle', mode = 'rb') as f:
        probs, grads, sum_grad = dill.load(f)
    scatter(X_tip, Y_tip, datotal[TIP], gmmpred[TIP], evaluation[TIP], probs, grads, sum_grad, dm = dm, save_img_dir=save_img_dir)
    
    with open(logdir + 'nn_res_shake.pickle', mode = 'rb') as f:
        probs, grads, sum_grad = dill.load(f)
    scatter_shake(X_shake, Y_shake, datotal[SHAKE], gmmpred[SHAKE], evaluation[SHAKE], probs, grads, sum_grad, save_img_dir=save_img_dir)
    
    optimize_check(dm, datotal, evaluation, logdir, save_img_dir=save_img_dir)


def Run(ct, *args):
    # i = args[0]
    # basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    # logdir = 'GMM12Sig8LCB4/checkpoints/t{}/ch500/'.format(i)
    # logdir = basedir + logdir
    # dm_path = logdir + 'dm.pickle'
    # with open(dm_path, mode="rb") as f:
    #     dm = dill.load(f)
    
    # X, Y = extract_trainig_data(dm)
    
    # # train_test(X, Y, 20)
    # # compute_grad_test(dm, 20)
    
    # model = BinaryClassification()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # model, max_acc = train(model, X, Y, criterion, optimizer, batch_size = len(X), n_epoch = 200)
    # probs, grads, sum_grad = compute_grad(dm, model)
    
    # with open(logdir + 'nn_res.pickle', mode = 'wb') as f:
    #     pickle.dump([probs, grads, sum_grad], f)
    # with open(logdir + 'nn_res.pickle', mode = 'rb') as f:
    #     probs, grads, sum_grad = dill.load(f)
    
    # # nn_res_concat = []
    # # for _ in range(20):
    # #     model = BinaryClassification()
    # #     criterion = nn.CrossEntropyLoss()
    # #     optimizer = optim.Adam(model.parameters(), lr=0.01)
    # #     model, max_acc = train(model, X, Y, criterion, optimizer, batch_size = len(X), n_epoch = 1000)
    # #     probs, grads, sum_grad = compute_grad(dm, model)
    # #     nn_res_concat.append([probs, grads, sum_grad])
    
    # # with open(logdir + 'nn_res_cocnat.pickle', mode = 'wb') as f:
    # #     pickle.dump(nn_res_concat, f)
    # # with open(logdir + 'nn_res_cocnat.pickle', mode = 'rb') as f:
    # #     nn_res_concat = dill.load(f)
    
    # # nn_res_concat = []
    # # models = []
    # # for _ in range(20):
    # #     model = BinaryClassification()
    # #     criterion = nn.CrossEntropyLoss()
    # #     optimizer = optim.Adam(model.parameters(), lr=0.01)
    # #     model, max_acc = train(model, X, Y, criterion, optimizer, batch_size = len(X), n_epoch = 100)
    # #     probs, grads, sum_grad = compute_grad(dm, model)
    # #     nn_res_concat.append([probs, grads, sum_grad])
    # #     models.append(model)
    # # probs, grads, sum_grad = compute_grad_mean(dm, models)
        
    # # with open(logdir + 'nn_res_cocnat.pickle', mode = 'wb') as f:
    # #     pickle.dump(nn_res_concat, f)
    # # with open(logdir + 'nn_res_mean.pickle', mode = 'wb') as f:
    # #     pickle.dump([probs, grads, sum_grad], f)
    # # with open(logdir + 'nn_res_cocnat.pickle', mode = 'rb') as f:
    # #     nn_res_concat = dill.load(f)
    # # with open(logdir + 'nn_res_mean.pickle', mode = 'rb') as f:
    # #     mean_probs, grads, mean_sum_grad = dill.load(f)
        
    # datotal, gmmpred, evaluation = setup_full(dm, logdir, recreate=False)
    # X = X.to('cpu').detach().numpy().copy()
    # Y = Y.to('cpu').detach().numpy().copy()
    # # scatter(X, Y, datotal[TIP], gmmpred[TIP], evaluation[TIP], probs, grads, sum_grad)
    # # scatter(X, Y, datotal[TIP], evaluation[TIP], None, None, None, nn_res_concat)
    # # scatter(X, Y, datotal[TIP], gmmpred[TIP], evaluation[TIP], nn_res_concat[-1][0], nn_res_concat[-1][1], nn_res_concat[-1][2], nn_res_concat, mean_res = [mean_probs, mean_sum_grad])
    # # datotal, gmmpred, evaluation = setup_full(dm, dm_path, recreate=False)
    
    # # eval_tip(datotal, dm, mode='Er', gmmpred=None, sum_grad=None, logdir=logdir, update = True)
    # # eval_tip(datotal, dm, mode='LCB', gmmpred=None, sum_grad=None, logdir=logdir, update = True)
    # # eval_tip(datotal, dm, mode='GMM', gmmpred=gmmpred, sum_grad=None, logdir=logdir, update = True)
    # # eval_tip(datotal, dm, mode='Edge', gmmpred=None, sum_grad=sum_grad, logdir=logdir, update = True)
    # eval_tip(datotal, dm, mode='Edge_GMM', gmmpred=gmmpred, sum_grad=sum_grad, logdir=logdir, update = True)
    
    # # eval_shake(datotal, dm, mode='Er', gmmpred=None, sum_grad=None, logdir=logdir, update = True)
    # # eval_shake(datotal, dm, mode='LCB', gmmpred=None, sum_grad=None, logdir=logdir, update = True)
    # eval_shake(datotal, dm, mode='GMM', gmmpred=gmmpred, sum_grad=None, logdir=logdir, update = True)
    # # eval_shake(datotal, dm, mode='Edge', gmmpred=None, sum_grad=sum_grad, logdir=logdir, update = True)
    
    # # datotal, gmmpred, evaluation = setup_full(dm, dm_path, recreate=False)
    # # optimize_check(dm, datotal, logdir)
    
    # for i in range(1,100):
    #     train_check("ErLCB4/checkpoints", i)
        
        # basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
        # logdir = 'GMM12Sig8LCB4/checkpoints/t{}/ch500/'.format(i)
        # logdir = basedir + logdir
        # dm_path = logdir + 'dm.pickle'
        # with open(dm_path, mode="rb") as f:
        #     dm = dill.load(f)
        # dm.LCB_ratio = 2
        # datotal, gmmpred, evaluation = setup_full(dm, dm_path, recreate=False)
        # eval_tip(datotal, dm, mode='GMM', gmmpred=gmmpred, sum_grad=None, logdir=logdir, update = True)
        # eval_shake(datotal, dm, mode='GMM', gmmpred=gmmpred, sum_grad=None, logdir=logdir, update = True)
        # tip_lcb = eval_tip(datotal, dm, mode='LCB', logdir=logdir)
    # train_check("GMM12Sig8LCB4/checkpoints", 4, 'ch500/')
    for i in range(1,100):
        train_check("GMM12Sig8LCB4/checkpoints", i, 'ch500/')
        check("GMM12Sig8LCB4/checkpoints", i, 'ch500/')
    
    opttest_comp_use_edge("GMM12Sig8LCB4/checkpoints", 99, ch='ch500/', ver = 2, use_edge=True, without_gmm=True, suffix='2')
    # opttest_comp_use_edge("GMM12Sig8LCB4/checkpoints", 99, ch='ch500/', ver = 2, use_edge=True, without_gmm=False)
    # opttest_comp_use_edge("GMM12Sig8LCB4/checkpoints", 99, ch='ch500/', ver = 2, use_edge=False)
    # opttest_comp_use_edge("ErLCB4/checkpoints", 99, ch='ch500/', ver = 2, use_edge=True)
    
    # check("GMM12Sig8LCB4/checkpoints", 3, 'ch500/')