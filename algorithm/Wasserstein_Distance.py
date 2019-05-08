import numpy as np
from scipy.stats import wasserstein_distance

# generate the px and py
x = [np.random.randint(1,11) for i in range(10)]
sum_x = np.sum(x)
px = x / sum_x
print("px value: {}".format(px))

y = [np.random.randint(1,11) for i in range(10)]
sum_y = np.sum(y)
py = y / sum_y
print("py value: {}".format(py))

print("scipy wasserstein_distance {}".format(wasserstein_distance(px,py)))


print("measure the JS and Wasserstein Distance ...")
from algorithm.JS_Distance import scipy_js
a = np.array([0.4,0.6,0,0])
b = np.array([0,0,0.7,0.3])
print("orthogonal vector *****************************************")
print("JS distatnce: {}".format(scipy_js(a,b)))
print("Log 2 :{}".format(np.log(2)))
print("Wasserstein distance: {}".format(wasserstein_distance(a,b)))

a = np.array([0.4,0.6,0,0])
b = np.array([0.4,0.6,0,0])
print("the same vector *****************************************")
print("JS distatnce: {}".format(scipy_js(a,b)))
print("Log 2 :{}".format(np.log(2)))
print("Wasserstein distance: {}".format(wasserstein_distance(a,b)))

a = np.random.uniform(0,1,50000)
b = np.random.uniform(0,1,50000)
print("the same vector *****************************************")
print("JS distatnce: {}".format(scipy_js(a,b)))
print("Log 2 :{}".format(np.log(2)))
print("Wasserstein distance: {}".format(wasserstein_distance(a,b)))