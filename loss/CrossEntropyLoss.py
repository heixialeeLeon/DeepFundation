import numpy as np
import torch
import torch.nn as nn

def PCrossEntropyLoss(x,y, average=True):
    lst = []
    for k in range(len(x)):
        softmax_score = np.exp(x[k][y[k]]) / np.exp(x[k]).sum()
        log_score = -np.log(softmax_score)
        lst.append(log_score)
    if average:
        return np.mean(lst)
    else:
        return np.sum(lst)

if __name__ == "__main__":
    x = torch.randn(2,4)
    y = torch.LongTensor(2).random_(4)
    print("Pytorch CrossEntropy loss: {}".format(nn.CrossEntropyLoss()(x,y)))
    print("Python CrossEntropy loss: {}".format(PCrossEntropyLoss(x.numpy(),y.numpy())))

    print("Pytorch CrossEntropy loss: {}".format(nn.CrossEntropyLoss(reduction="sum")(x, y).numpy()))
    print("Python CrossEntropy loss: {}".format(PCrossEntropyLoss(x, y, False)))

