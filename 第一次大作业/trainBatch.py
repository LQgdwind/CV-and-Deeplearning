import dataDisplay as dd
import torch
from torch import nn

def accuracy(y_hat, y):
    y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

def top5_accuracy(y_hat, y):
    y_hat1 = y_hat.sort(1, True)[0][:][0:5]
    cmp = (astype(y_hat1[0], y.dtype) == y) | (astype(y_hat1[1], y.dtype) == y) | (astype(y_hat1[2], y.dtype) == y) | (astype(y_hat1[3], y.dtype) == y) | (astype(y_hat1[4], y.dtype) == y)
    return float(reduce_sum(astype(cmp, y.dtype)))

def evaluate_accuracy_gpu(net, data_iter, device=None ,mode = 1):
    if mode == 1:
        if isinstance(net, nn.Module):
            net.eval()
            if not device:
                device = next(iter(net.parameters())).device
        metric = dd.Accumulator(2)

        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)

                metric.add(accuracy(net(X), y), size(y))
        return metric[0] / metric[1]
    elif mode == 5:
        if isinstance(net, torch.nn.Module):
            net.eval()
        metric = dd.Accumulator(2)

        with torch.no_grad():
            for X, y in data_iter:
                metric.add(top5_accuracy(net(X), y), size(y))
        return metric[0] / metric[1]

def train_batch(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# 一些数学工具的定义
ones = torch.ones
zeros = torch.zeros
tensor = torch.tensor
arange = torch.arange
meshgrid = torch.meshgrid
sin = torch.sin
sinh = torch.sinh
cos = torch.cos
cosh = torch.cosh
tanh = torch.tanh
linspace = torch.linspace
exp = torch.exp
log = torch.log
normal = torch.normal
rand = torch.rand
matmul = torch.matmul
int32 = torch.int32
float32 = torch.float32
concat = torch.cat
stack = torch.stack
abs = torch.abs
eye = torch.eye
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
