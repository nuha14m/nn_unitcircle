 #iter-tools
import operator
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np

def step(x):
    if x>0: return 1
    else: return 0

def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))

def perceptron(A, w, b, x):
    return A(dot(w,x)+b)

def dot(w,x):
    sum=0
    for i in range(len(w)):
        sum+= w[i]*x[i]
    return sum

def binarycon(n, lfinal):
    n1 = str(bin(n))
    n1=n1[2:]
    if lfinal> len(n1): n1 = '0'*int(lfinal-len(n1))+n1
    return n1

def train(n, num_input):
    n = binarycon(n, int(math.pow(2,num_input)))
    b=0
    w=(0,)*num_input
    train_set= truth_table(num_input, n)
    x_1=tuple()
    for i in range(100):
        for key in train_set.keys():
            x,y= key, train_set[key]
            y_1 = step(dot(w,x)+b)
            err = int(y)-y_1
            x_1 = (int(i) * err for i in x)
            w=  tuple(map(sum,zip(w,x_1)))
            b+=err
    return w,b
   
def circle_perceptron(x,y):
    A= perceptron(sigmoid, (0,1), 1, (x,y))
    B= perceptron(sigmoid, (0,-1), 1, (x,y))
    C= perceptron(sigmoid, (-1,0), 1, (x,y))
    D= perceptron(sigmoid, (1,0), 1, (x,y))
    E= perceptron(sigmoid, (1,1,1,1), -3.5, (A,B,C,D))
    return E

def circleeqn(x,y):
    return math.pow(x,2)+math.pow(y,2)
 
if __name__ == "__main__":
    threshold=0.3399
    
    for x in np.arange(-20, 20):
        for y in np.arange(-20, 20):
            P= circle_perceptron(x*0.1,y*0.1)
            val=circleeqn(x*0.1,y*0.1)
            if P<=threshold and val>1: plt.plot([x*0.1], [y*0.1],'bo', markersize=2)
            elif P<=threshold and val<=1: plt.plot([x*0.1], [y*0.1],'yo', markersize=2)
            elif P>threshold and val<=1: plt.plot([x*0.1], [y*0.1],'go', markersize=2)
            elif P>threshold and val>1: plt.plot([x*0.1], [y*0.1], 'co', markersize=2)

    plt.show()
       
