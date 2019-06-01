import operator
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import random

def dfsigmoid(x):
    return (pow(math.e, -x))/(pow(1+pow(math.e, -x), 2))

def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))

def random_vector():
    w=tuple()
    for i in range(19):
        w+=tuple([random.random()*2-1])
    return w

def E(X, ww):
    error=0
    rt=sigmoid(ww[17],1)
    for val in X:
        if circleeqn(val[0],val[1])<=1: y=1
        else: y=0
        if c_percep(val[0],val[1],ww)<=rt:myx=1
        else: myx=0
        error+= math.pow(y-myx,2)
    return error

def back_prop_matrix(training_set):
    A = np.vectorize(sigmoid)
    dA = np.vectorize(dfsigmoid)
    a,b,W, delta= dict(), dict(), dict(), dict()
    N=2
    lmda = 0.5
    error=50
    while(error>=50): #intialize random weight matrices
        W[0], W[1] = np.random.rand(2,4), np.random.rand(4,1)
        b[1], b[2] = np.random.rand(1,4), np.random.rand(1,1)
        for i in range(3000):#each epoch do:
            err = 0
            for (x,y) in training_set:
                a[0]=x
                for L in range(1, N+1):
                    dot_L = a[L-1] * W[L-1] + b[L]
                    a[L] = A(dot_L)
                delta[N] = np.multiply(dA(a[1]*W[1]+b[2]), y-a[2])
                for L in range(N-1, 0, -1):
                    delta[L] = np.multiply(dA(a[L-1]*W[L-1]+b[L]),delta[L+1]*W[L].transpose())
                for L in range(N): # update weights
                    W[L] = W[L] + (lmda*a[L].transpose()*delta[L+1])
                    b[L+1]= b[L+1] + (lmda*delta[L+1])
                if y is 0 and a[2]>0.5: err+=1
                if y is 1 and a[2]<=0.5: err+=1
            if i%10==0: print(i, err, b, W)
            error= err
            lmda = err/10000
    #print(y, a[2], '0.5', 0.5* np.linalg.norm(y-a[2])**2)
    return a, b, W

def circleeqn(x,y):
    return math.pow(x,2)+math.pow(y,2)

training_set = []
with open("10000_pairs.txt", "r") as f:
    for line in f:
        xs, ys = line.split()
        x, y = float(xs), float(ys)
        answer = 1 if x**2 + y**2 <= 1 else 0
        training_set.append((np.matrix([x, y]), answer))

a, b, W = back_prop_matrix(training_set)

