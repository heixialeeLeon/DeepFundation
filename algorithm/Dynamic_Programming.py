#  https://blog.csdn.net/u013309870/article/details/75193592
import numpy as np
price =[0,1,5,8,9,10,17,17,20,24,30]

# 钢铁分段问题
def max_price(n):
    max_price = np.zeros((n+1,1))
    for i in range(n+1):
        current = 0
        for j in range(i):
            current = max(current,max_price[j]+price[i-j])
        max_price[i] = current
    for i in range(n+1):
        print(max_price[i])

# 背包问题
# https://www.cnblogs.com/Christal-R/p/Dynamic_programming.html

volume = [0,2,3,4,5]
price =  [0,3,4,5,6]
capacity = 8
V = np.zeros(shape=(len(volume),capacity+1))

def findMax():
    for i in range(1,len(volume)):
        for j in range(1,capacity+1):
            if j < volume[i]:
                V[i][j] = V[i-1][j]
            else:
                value = V[i-1][j-volume[i]] + price[i]
                V[i][j] = max(V[i-1][j],value)
    for i in range(len(volume)):
        print(V[i])


if __name__ == "__main__":
    # max_price(10)
    findMax()