import torch
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing

def sigmoid(x):
    x = 1/(1+np.exp(-x))
    return x

def pzero_score_normalize(x):
    mean = x.mean()
    std = x.std(ddof=1)
    x = (x-mean)/std
    return x

def pmin_max_normalize(x):
    min = np.min(x)
    max = np.max(x)
    return (x-min)/(max-min)

def sklean_normalize(x):
    x = x.reshape(-1,1)
    x = preprocessing.normalize(x.reshape(-1, 1), axis=0, norm='l2')
    return x.reshape(-1)

def pnorm2(x):
    val=0.0
    for item in x:
        val += item*item
    return np.sqrt(val)

def pnormalize(x):
    #norm = np.linalg.norm(x,axis=0)
    norm = pnorm2(x)
    return x / norm

x = np.array([1,3,4,5])
tensor_x = torch.from_numpy(x).float()

print("******************************************************************")
print("pnorm2 {}".format(pnorm2(x)))
print("np.linalg.norm {}".format(np.linalg.norm(x,axis=0)))

print("******************************************************************")
print("sigmoid: {}".format(sigmoid(x)))
print("zero-score normalize ： {}".format(pzero_score_normalize(x)))
print("min-max normalize ： {}".format(pmin_max_normalize(x)))

# print("np.linalg.norm: {}".format(preprocessing.normalize(x, norm='l2'))
print("sklearn normalize: {}".format(sklean_normalize(x)))
print("torch normalize : {}".format(F.normalize(tensor_x,dim=0).numpy()))
print("pnormalize {}".format(pnormalize(x)))