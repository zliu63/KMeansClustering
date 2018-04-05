from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
#import dataset
myMap = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
filename = "./data/iris.data"
lines = [line.rstrip('\n').split(",") for line in open(filename,"r")] 
lines.pop(0)
data = np.array(lines)
X = data[:,0:4].astype(np.float)
Y = data[:,4]
N = Y.shape[0]
for i in range(N):
    Y[i] = myMap[Y[i]]
Y = Y.astype(np.float)
# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)

def choose(x,C_curr):
    dist = np.square(C_curr-x)
    dist = np.sum(dist,axis = 1)
    return np.argmin(dist)

def k_means(C):
    '''
    Repeat until convergence:
        2.1 (Recenter.) Set μj := mean(Cj) for j ∈ (1,...,k).
        2.2 (Reassign). Update Cj := {xi : μ(xi) = μj} for j ∈ (1,...,k)
        (break ties arbitrarily).
    '''
    pred = np.zeros(N)
    C_curr = np.array(C)
    K = C_curr.shape[0]
    D = C_curr.shape[1]
    C_next = np.zeros((K,D))
    for i in range(N):
        pred[i] = choose(X[i],C_curr)
    for k in range(K):
        C_next[k] = np.mean(X[pred==k],axis = 0)
    while not np.array_equal(C_next,C_curr):
        C_curr = deepcopy(C_next)
        for i in range(N):
            pred[i] = choose(X[i],C_curr)
        for k in range(K):
            C_next[k] = np.mean(X[pred==k],axis = 0)            
    return C_next








