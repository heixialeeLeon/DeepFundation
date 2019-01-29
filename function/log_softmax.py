import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def getSoftDim(x):
    ndim = len(x.shape)
    if ndim == 0 or ndim == 1 or ndim == 3:
        return 0
    else:
        return 1

def PLogSoftmax(x, dim=None):
    if dim is None:
        dim = getSoftDim(x)
    softmax_score = np.exp(x) / np.exp(x).sum(axis=dim, keepdims=True)
    log_score = np.log(softmax_score)
    return log_score

if __name__ == "__main__":
    data = torch.randn(1,3,200,300)
    output = F.log_softmax(data)
    output_p = PLogSoftmax(data.numpy())
    print(output.sum())
    print(output_p.sum())

    output = F.log_softmax(data, dim=2)
    output_p = PLogSoftmax(data.numpy(), dim=2)
    print(output.mean())
    print(output_p.mean())

    data =torch.randn(2,3)
    output = F.log_softmax(data)
    output_p = PLogSoftmax(data.numpy())
    print(output.mean())
    print(output_p.mean())

    data = torch.randn(3,20,40)
    output = F.log_softmax(data)
    output_p = PLogSoftmax(data.numpy())
    print(output.mean())
    print(output_p.mean())