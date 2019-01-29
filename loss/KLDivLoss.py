import numpy as np
import torch
import torch.nn as nn

def PKLDivLoss(x,y,average=True):
    y_log = y.copy()
    y_mask = y.copy()
    y_mask[y_mask<=0] =0
    y_log[y_log<=0] =0
    y_log[y_log>0] = np.log(y_log[y_log>0])
    c = y_mask*(y_log-x)
    if average:
        return np.mean(c)
    else:
        return c
if __name__ == "__main__":
    x = torch.randn(2,3)
    y = torch.randn(2,3)

    print(y)
    print("**********************************************************")
    print("Pytorch KLDivLoss: {}".format(nn.KLDivLoss()(x, y)))
    print("Python PKLDivLoss: {}".format(PKLDivLoss(x.numpy(), y.numpy())))
    print("**********************************************************")
    print("Pytorch KLDivLoss: {}".format(nn.KLDivLoss(reduction='none')(x, y)))
    print("Python PKLDivLoss: {}".format(PKLDivLoss(x.numpy(), y.numpy(),average=False)))