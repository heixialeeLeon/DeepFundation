import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def PBatchNorm2d(data):
    mean = np.mean(data, axis=(0,2,3))
    var = np.var(data,axis=(0,2,3))
    data_norm = (data -mean)/np.sqrt(var+1e-8)
    return data_norm

def test():
    m = nn.BatchNorm2d(3,momentum=1)
    input = torch.randn(4,3,2,2)
    output = m(input)
    print("input: {}, output: {}".format(input.shape, output.shape))

    a =(input[0, 0, :, :] + input[1, 0, :, :] + input[2, 0, :, :] + input[3, 0, :, :]).sum() / 16
    b =(input[0, 1, :, :] + input[1, 1, :, :] + input[2, 1, :, :] + input[3, 1, :, :]).sum() / 16
    c = (input[0, 2, :, :] + input[1, 2, :, :] + input[2, 2, :, :] + input[3, 2, :, :]).sum() / 16
    print("channel 0 :{}, channel 1 :{}, channel 2:{}".format(a.data,b.data,c.data))
    print(m.running_mean.data[0], m.running_mean.data[1], m.running_mean.data[2])
    print(m)


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

    test()