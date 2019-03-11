import numpy as np

a1 = np.array([0,0,5,5])
a2 = np.array([2,2,3,3])
print("方差：")
print(a1.var(), a2.var())
print("标准偏差")
print(a1.std(), a2.std())
print("均值")
print(a1.mean(), a2.mean())