import numpy as np
import matplotlib.pyplot as plt
import random

rate = 0.1 # learning rate
#  模拟数据
x = [30	,35,37,	59,	70,	76,	88,	100]
y = [1100,	1423,	1377,	1800,	2304,	2588,	3495,	4839]
# 初始化a,b值
a = 10.0
b = -20.0

def da(y,y_p,x):
    return (y-y_p)*(-x)

def db(y,y_p):
    return (y-y_p)*(-1)

def calc_loss(a,b,x,y):
    tmp = y - (a * x + b)
    tmp = tmp ** 2  # 对矩阵内的每一个元素平方
    SSE = sum(tmp) / (2 * len(x))
    return SSE

def shuffle_data(x,y):
    # 随机打乱x，y的数据，并且保持x和y一一对应
    seed = random.random()
    random.seed(seed)
    random.shuffle(x)
    random.seed(seed)
    random.shuffle(y)
# 数据归一化
x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)

for i in range(0,len(x)):
    x[i] = (x[i] - x_min)/(x_max - x_min)
    y[i] = (y[i] - y_min)/(y_max - y_min)

all_loss = []
all_step = []
va = 0
vb = 0
gama = 0.9
for step in range(1,200):
    loss = 0
    all_da = 0
    all_db = 0
    shuffle_data(x, y)
    for i in range(0,len(x)):
        y_p = a*x[i] + b
        loss = loss + (y[i] - y_p)*(y[i] - y_p)/2
        all_da = all_da + da(y[i],y_p,x[i])
        all_db = all_db + db(y[i],y_p)

    loss = loss / len(x)
    all_da = all_da / len(x)
    all_db = all_db / len(x)

    va = gama*va + rate*all_da
    vb = gama*vb + rate*all_db
    a = a - va
    b = b - vb

    # 绘制图3中的回归直线
    plt.subplot(2, 1, 2)
    plt.plot(x, y)
    plt.plot(x, y, 'o')
    x_ = np.linspace(0, 1, 2)
    y_draw = a * x_ + b
    plt.plot(x_, y_draw)
    # 绘制图4的loss更新曲线
    all_loss.append(loss)
    all_step.append(step)
    plt.subplot(2,1,1)
    plt.plot(all_step,all_loss,color='orange')
    plt.xlabel("step")
    plt.ylabel("loss")

    if step%1 == 0:
        print("step: ", step, " loss: ", loss)
        plt.pause(0.1)
plt.pause(99999999999)