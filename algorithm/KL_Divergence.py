import numpy as np
import scipy.stats

def PKL(x,y):
    assert len(x) == len(y)
    KL = 0.0
    for i in range(len(x)):
        KL +=x[i] * np.log(x[i]/y[i])
    return KL

# generate the px and py
x = [np.random.randint(1,11) for i in range(10)]
sum_x = np.sum(x)
px = x / sum_x
print("px value: {}".format(px))

y = [np.random.randint(1,11) for i in range(10)]
sum_y = np.sum(y)
py = y / sum_y
print("py value: {}".format(py))

print("scipy KL value: {}".format(scipy.stats.entropy(x,y)))
print("python KL value: {}".format(PKL(px,py)))