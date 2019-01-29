import numpy as np
import torch
import torch.nn as nn

def PMSELoss(x,y, average=True):
    if average:
        return ((x-y)**2).mean()
    else:
        return ((x-y)**2).sum()

if __name__ == "__main__":
    x = torch.randn(2,3)
    y = torch.randn(2,3)
    print("Pytorch MSE loss: {}".format(nn.MSELoss()(x,y)))
    print("Python MSE loss: {}".format(PMSELoss(x.numpy(),y.numpy())))

    print("Pytorch MSE loss: {}".format(nn.MSELoss(reduction="sum")(x, y).numpy()))
    print("Python MSE loss: {}".format(PMSELoss(x.numpy(), y.numpy(), False)))