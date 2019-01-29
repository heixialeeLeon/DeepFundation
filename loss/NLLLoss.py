import numpy as np
import torch
import torch.nn as nn
from function.log_softmax import PLogSoftmax,getSoftDim

def PNLLLoss(x, y, dim=None):
    if dim is None:
        dim = getSoftDim(x)
    #mask = np.zeros_like(x)
    dim_len = x.shape[dim]
    label = np.expand_dims(y,dim).repeat(dim_len,dim)
    c = np.arange(dim_len)
    axis = [1]*len(x.shape)
    axis[dim] = -1
    c = c.reshape(*axis)
    index = c == label
    value = -(x*index)
    return np.sum(value,axis=dim).mean()

def PNLLLoss2(x,y, average=True):
    lst = []
    for k in range(len(x)):
        lst.append(-x[k][y[k]])
    if(average):
        return np.mean(lst)
    else:
        return lst

if __name__ == "__main__":
    x = torch.randn(2,4)
    y = nn.LogSoftmax(dim=1)(x)
    print("Pytorch LogSoftmax: {}".format(y))
    print("Python LogSoftmax : {}".format(PLogSoftmax(x.numpy())))

    x0 = torch.randn(3,4)
    x = nn.LogSoftmax(dim=1)(x0)
    y = torch.LongTensor(3).random_(4)
    print("NLLLoss: >>>>>>>>>>>>>>>>>>>>>>>>")
    print("Pytorch NLLLoss : {}".format(nn.NLLLoss()(x,y).numpy()))
    print("Pytorch CrossEntropy loss: {}".format(nn.CrossEntropyLoss()(x, y)))
    print("Python NLLLoss : {}".format(PNLLLoss2(x.numpy(), y.numpy())))
    print("Python NLLLoss : {}".format(PNLLLoss(x.numpy(), y.numpy())))

    print("NLLLoss ndim=3: >>>>>>>>>>>>>>>>>>>>>>>>")
    x0 = torch.randn(3, 20, 30)
    x = nn.LogSoftmax(dim=0)(x0)
    y = torch.empty(20, 30, dtype=torch.long).random_(0, 3)
    print("Python NLLLoss : {}".format(PNLLLoss(x.numpy(), y.numpy())))

    print("NLLLoss ndim=4: >>>>>>>>>>>>>>>>>>>>>>>>")
    x0 = torch.randn(4,2,20,30)
    x = nn.LogSoftmax(dim=1)(x0)
    y = torch.empty(4,20,30, dtype =torch.long).random_(0,2)
    print("Pytorch NLLLoss : {}".format(nn.NLLLoss()(x, y).numpy()))
    print("Pytorch CrossEntropy loss: {}".format(nn.CrossEntropyLoss()(x, y)))
    print("Python NLLLoss : {}".format(PNLLLoss(x.numpy(), y.numpy())))

    a = np.arange(12)
    print(a.shape)
    print(a.ndim)
    a.ndim=3
    print(a.ndim)