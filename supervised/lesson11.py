import math
import numpy as np
from scipy.spatial.distance import cityblock
from operator import itemgetter


def d(x, q):
    x1 = x[0] - q[0]
    y1 = x[1] - q[1]
    return math.sqrt(x1 * x1 + y1 * y1)


def m(x, q):
    return cityblock(x, q)


x = [[1, 6,  7.0],
     [2, 4,  8.0],
     [3, 7, 16.0],
     [6, 8, 44.0],
     [7, 1, 50.0],
     [8, 4, 68.0]
     ]

y = [7, 8, 16, 44, 50, 68]


def e(x, q):
    r = []
    for p in x:
        r.append([d(p[0:2], q),p[2]])
    r.sort(key=itemgetter(0))
    return np.array(r)


def f(x, q):
    r = []
    for p in x:
        r.append([m(p[0:2], q), p[2]])
    r.sort(key=itemgetter(0))
    return np.array(r)


q = [4, 2]

print(x)

r = e(x,q)
print(r)
k3 = r[:,1]
k3 = k3[0:3]
print(k3)
print(np.mean(k3))

r = f(x,q)
print(r)
k3 = r[:,1]
k3 = k3[0:4]
print(k3)
print(np.mean(k3))

print(np.mean([8]))
print(np.mean([8,50]))