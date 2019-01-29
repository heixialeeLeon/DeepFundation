import numpy as np
import torch
import torch.nn as nn

def PL1Loss(x,y, average=True):
    if average:
        return abs(x-y).mean()
    else:
        return abs(x-y).sum()

if __name__ == "__main__":
    x = torch.randn(2,3)
    y = torch.randn(2,3)
    print("Pytorch L1 loss: {}".format(nn.L1Loss()(x,y)))
    print("Python L1 loss: {}".format(PL1Loss(x.numpy(),y.numpy())))

    print("Pytorch L1 loss: {}".format(nn.L1Loss(reduction="sum")(x, y).numpy()))
    print("Python L1 loss: {}".format(PL1Loss(x.numpy(), y.numpy(), False)))
