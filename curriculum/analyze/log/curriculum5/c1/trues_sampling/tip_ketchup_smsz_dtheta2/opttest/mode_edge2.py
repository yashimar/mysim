from math import gamma
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


def extract_trainig_data(dm):
    log = dm.log
    X = []
    Y = []
    for i in log['ep']:
        if log['skill'][i] == 'tip':
            x = []
            x.append(log['est_optparam'][i])
            x.append(log['smsz'][i])
            r = log['r_at_est_optparam'][i]
            c = 1 if r <-1 else 0
            X.append(x)
            Y.append(c)
    
    X = torch.tensor(X, dtype = torch.float32)
    Y = torch.tensor(Y, dtype = torch.int64)
    
    return X, Y


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        n_units = 512
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, n_units),
            nn.ReLU(),
            nn.BatchNorm1d(n_units),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.BatchNorm1d(n_units),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.BatchNorm1d(n_units),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.BatchNorm1d(n_units),
            nn.Linear(n_units, n_units),
            nn.ReLU(),
            nn.BatchNorm1d(n_units),
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
    
    
def get_accuracy(y_prob, y_true):
    y_prob = y_prob.to('cpu')
    y_true = y_true.to('cpu')
    y_prob = torch.argmax(y_prob, axis = 1)
    return (1. * sum(y_prob == y_true) / len(y_true)).item()


def train(model, X, Y, criterion, optimizer, batch_size = 10, n_epoch = 100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    torch.backends.cudnn.benchmark = True

    dataset = Dataset(X, Y)
    dataloader = DataLoader(dataset, batch_size = batch_size)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min')
    # scheduler = ReduceLROnPlateau(optimizer, 'max')
    
    model.to(device)
    model.train()

    max_acc = 0
    for epoch in range(1, n_epoch+1):
        epoch_loss = 0
        epoch_acc = 0
        epoch_data_len = 0
        for batch, (bX, bY) in enumerate(dataloader):
            bX, bY = bX.to(device), bY.to(device)

            pred = model(bX)
            loss = criterion(pred, bY)
            acc = get_accuracy(pred, bY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss
            epoch_acc += acc*len(bX)
            epoch_data_len += len(bX)

        epoch_loss /= epoch_data_len
        epoch_acc /= epoch_data_len
        max_acc = max(max_acc, epoch_acc)
        print('Epoch {}: | Loss: {:.5f} | Acc: {:.5f} | Max Acc: {:.5f}'.format(epoch, epoch_loss, epoch_acc, max_acc))
        
        scheduler.step(epoch_loss)
        # scheduler.step(epoch_acc)
        if epoch_acc == 1:
            break

    return model, max_acc


def train_test(X, Y, n_trial):
    dt_list = []
    acc_list = []
    for _ in range(n_trial):
        model = BinaryClassification()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        t1 = time()
        model, max_acc = train(model, X, Y, criterion, optimizer, batch_size = 60, n_epoch = 400)
        t2 = time()
        dt = t2 - t1
        dt_list.append(dt)
        acc_list.append(max_acc)
    print(np.mean(dt_list), np.std(dt_list))
    print(np.mean(acc_list), np.std(acc_list))
    
    
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
    
    
def scatter(preds, grads1, grads2):
    # グラフ1 サンプル点をプロット
    # ラベル1: red, ラベル0: black
    # グラフ2 preds のヒートマップとサンプル点
    # 0 ~ 1: white ~ red
    # グラフ3 grads1 の勾配
    # グラフ4 grads2 の勾配
    # グラフ5 |grads1| + |grads2| の勾配
    pass
        

def Run(ct, *args):
    basedir = "/home/yashima/ros_ws/ay_tools/ay_skill_extra/mysim/curriculum/analyze/log/curriculum5/c1/trues_sampling/tip_ketchup_smsz_dtheta2/opttest/logs/onpolicy2/"
    logdir = 'GMM12Sig8LCB4/checkpoints/t1/ch500/'
    path = basedir+logdir+'dm.pickle'
    with open(path, mode="rb") as f:
        dm = dill.load(f)
    
    X, Y = extract_trainig_data(dm)
    
    # train_test(X, Y, 20)
    compute_grad_test(dm, 20)
    
    
    