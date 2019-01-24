import numpy as np

def sigmoid(x):
    score =  [ 1/float(1+np.exp(-item)) for item in x ]
    return score

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def cross_entropy(x,y):
    m = y.shape[0]
    p = softmax(x)
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood)/m
    return loss

if __name__ == "__main__":
    input = np.arange(1,5)
    label = np.array([1])
    print("input is: {}".format(input))
    print("sigmoid is ：{}".format(sigmoid(input)))
    print("softmax is : {}".format(softmax(input)))
    input = np.arange(1,9).reshape(2,4)
    label = np.array([1,2]).transpose()
    print("cross_entropy is : {}".format(cross_entropy(input,label)))