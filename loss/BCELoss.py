import numpy as np
import torch
import torch.nn as nn

def PBCELoss(x, y, average = True):
    c = x.copy()
    c[y == 1] = -np.log(c[y==1])
    c[y == 0] = -np.log(1-c[y==0])
    if average:
        return np.mean(c)
    else:
        return c

if __name__ == "__main__":
    x0 = torch.randn(3)
    x = nn.Sigmoid()(x0)
    y = torch.FloatTensor(3).random_(2)

    print("x: {}".format(x))
    print("y: {}".format(y))
    print("**********************************************************")
    print("Pytorch BCELoss: {}".format(nn.BCELoss()(x, y)))
    print("Python PBCELoss: {}".format(PBCELoss(x.numpy(), y.numpy())))
    print("**********************************************************")
    print("Pytorch BCELoss: {}".format(nn.BCELoss(reduction='none')(x, y)))
    print("Python PBCELoss: {}".format(PBCELoss(x.numpy(), y.numpy(),average=False)))

    print("**********************************************************")
    x0 = torch.randn(3, 2)
    x = nn.Sigmoid()(x0)
    y = torch.FloatTensor(3, 2).random_(2)
    print("Pytorch BCELoss: {}".format(nn.BCELoss()(x, y)))
    print("Python PBCELoss: {}".format(PBCELoss(x.numpy(), y.numpy())))
    print("**********************************************************")
    print("Pytorch BCELoss: {}".format(nn.BCELoss(reduction='none')(x, y)))
    print("Python PBCELoss: {}".format(PBCELoss(x.numpy(), y.numpy(), average=False)))
