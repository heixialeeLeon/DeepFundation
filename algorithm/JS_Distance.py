import numpy as np
import scipy.stats
from scipy.spatial import distance

def PKL(x,y):
    assert len(x) == len(y)
    KL = 0.0
    for i in range(len(x)):
        KL +=x[i] * np.log(x[i]/y[i])
    return KL

def python_js(x,y):
    m = (x+y)/2
    js=0.5*PKL(x,m) + 0.5*PKL(y,m)
    return js

def scipy_js(x,y):
    m = (x+y)/2
    js = 0.5*scipy.stats.entropy(x,m)+0.5*scipy.stats.entropy(y,m)
    return js

if __name__ == "__main__":
    # generate the px and py
    x = [np.random.randint(1,11) for i in range(10)]
    sum_x = np.sum(x)
    px = x / sum_x
    print("px value: {}".format(px))

    y = [np.random.randint(1,11) for i in range(10)]
    sum_y = np.sum(y)
    py = y / sum_y
    print("py value: {}".format(py))

    print("scipy JS distance {}".format(scipy_js(px,py)))
    print("python JS distance {}".format(python_js(px,py)))