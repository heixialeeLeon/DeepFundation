import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def PBatchNorm2d(data):
    mean = np.mean(data, axis=(0,2,3))
    var = np.var(data,axis=(0,2,3))
    data_norm = (data -mean)/np.sqrt(var+1e-8)
    return data_norm


if __name__ == "__main__":
    data = torch.randn(2,1,3,3)
    print(data.numpy())
    print("*************************************************")
    f = nn.BatchNorm2d(1,affine=False)
    f.train()
    output = f(data)
    print(output)
    print("*************************************************")
    print(PBatchNorm2d(data.numpy()))