import numpy as np
import torch
import torch.nn as nn

def sterling_approx(x):
    return x*np.log(x) - x + 0.5*np.log(2*np.pi*x)

def sterling_approx_filter(x):
    y = x.copy()
    c1_index = y>1
    c2_index = y<=1
    c1 = sterling_approx(y[c1_index])
    y[c2_index] = 0
    y[c1_index] = c1
    return y

def sterling_approx_array(x):
    x[x <= 1] = 0
    x[x>1] = sterling_approx(x[x>1])
    return x

def PPoissonNLLLoss(x,y,average=True, full=False):
    if full:
        c = np.exp(x) - y*x + sterling_approx_array(y)
    else:
        c = np.exp(x) - y*x
    if average:
        return np.mean(c)
    else:
        return c


if __name__ == "__main__":
    x = torch.randn(2,4)
    y = torch.randn(2,4)
    print(y)
    print("**********************************************************")
    print("Pytorch PoissonNLLLoss: {}".format(nn.PoissonNLLLoss()(x,y)))
    print("Python  PoissonNLLLoss: {}".format(PPoissonNLLLoss(x.numpy(),y.numpy())))

    print("Pytorch PoissonNLLLoss full: {}".format(nn.PoissonNLLLoss(full=True)(x, y)))
    print("Python  PoissonNLLLoss full: {}".format(PPoissonNLLLoss(x.numpy(), y.numpy(), full=True)))

    print("Pytorch PoissonNLLLoss: {}".format(nn.PoissonNLLLoss(reduction='none', full=True)(x, y)))
    print("Python  PoissonNLLLoss: {}".format(PPoissonNLLLoss(x.numpy(), y.numpy(),average=False,full=True)))

    # print(sterling_approx_filter(y.numpy()))
    # print(sterling_approx_array(y.numpy()))


