import numpy as np


def lrn(w, x, b, r):
    b = b + r
    w = w + x * r
    return w, b, r


def pcp(w, x, b):
    return np.dot(w, x) + b


# w = np.array([3, 4])
# x = np.array([1, 1])
# b = -10
# r = 0.1
#
# for i in range(20):
#     w, b, r = lrn(w, x, b, r)
#     p = pcp(w, x, b)
#     print p, w, x, b, r
#

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def loss(w, x, b):
    p = pcp(w, x, b)
    s = sigmoid(p)
    return s


# w = [4, 5]
# b = -9
# x = [1, 1]
# print loss(w, [1, 1], b)
# print loss(w, [2, 4], b)
# print loss(w, [5, -5], b)
# print loss(w, [-4, 5], b)

def softmax(L):
    return np.max(np.exp(L))


l = np.array([1, 2, 3, 4])
print softmax(l)

# P = loss(P)
def cross_entropy1(Y, P):
    s = 0
    for i in range(len(Y)):
        v = Y[i] * np.log(P[i]) + (1.0 - Y[i]) * np.log(1.0 - P[i])
        s += v
    return -s/len(Y)


def cross_entropy(Y, P):
    s = 0
    for i in range(len(Y)):
        r = 0
        for j in range(len(P)):
            r += Y[i, j] * np.log(P[i, j])
        s += r
    return s / len(Y)

