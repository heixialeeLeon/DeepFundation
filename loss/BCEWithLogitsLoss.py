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

def sigmoid(x):
    x = 1/(1+np.exp(-x))
    return x

def PBCEWithLogitsLoss(x,y,average = True):
    x = sigmoid(x)
    return PBCELoss(x,y,average)

if __name__ == "__main__":
    x = torch.randn(3)
    xs = nn.Sigmoid()(x)
    p_xs = sigmoid(x.numpy())
    y = torch.FloatTensor(3).random_(2)

    print("x: {}".format(x))
    print("xs: {}".format(xs))
    print("p_xs: {}".format(p_xs))
    print("y: {}".format(y))
    print("**********************************************************")
    print("Pytorch BCELoss: {}".format(nn.BCELoss()(xs, y)))
    print("Pytorch BCEWithLogitsLoss: {}".format(nn.BCEWithLogitsLoss()(x, y)))
    print("Python PBCEWithLogitsLoss: {}".format(PBCEWithLogitsLoss(x.numpy(), y.numpy())))
    print("**********************************************************")
    print("Pytorch BCEWithLogitsLoss: {}".format(nn.BCEWithLogitsLoss(reduction='none')(x, y)))
    print("Python PBCEWithLogitsLoss: {}".format(PBCEWithLogitsLoss(x.numpy(), y.numpy(),average=False)))

    print("**********************************************************")
    x = torch.randn(3, 2)
    xs = nn.Sigmoid()(x)
    p_xs = sigmoid(x.numpy())
    y = torch.FloatTensor(3, 2).random_(2)
    print("Pytorch BCELoss: {}".format(nn.BCELoss()(xs, y)))
    print("Pytorch BCEWithLogitsLoss: {}".format(nn.BCEWithLogitsLoss()(x, y)))
    print("Python PBCEWithLogitsLoss: {}".format(PBCEWithLogitsLoss(x.numpy(), y.numpy())))
    print("**********************************************************")
    print("Pytorch BCEWithLogitsLoss: {}".format(nn.BCEWithLogitsLoss(reduction='none')(x, y)))
    print("Python PBCEWithLogitsLoss: {}".format(PBCEWithLogitsLoss(x.numpy(), y.numpy(), average=False)))