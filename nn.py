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

def truth_table(bits,n):
    t = dict()
    i=0
    for val in itertools.product('10', repeat=bits):
        t[val]=int(n[i])
        i+=1
    return t

def pretty_print(table):
    for val in table.keys():
        print(val,'|', table[val])
        
def dot(w,x):
    sum=0
    for i in range(len(w)):
        sum+= w[i]*x[i]
    return sum

def check(n,w,b,num_input):
    mytable= truth_table(len(w),n)
    count =0
    for val in mytable.keys():
        if perceptron(step,w,b,val)==int(mytable[val]):
            count+=1
    return float(count/len(n))

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
#return int(check(n,w,b, num_input))

def AND(x):
    return perceptron(step, (1,1), -1.5 ,x)
def NAND(x):
    return perceptron(step, (-1,-1), 1.5, x )
def OR(x):
    return perceptron(step, (1,1),-0.5, x)

def XOR(x):
    gate_1= NAND(x)
    gate_2= OR(x)
    new_x= np.array([gate_1, gate_2])
    output= AND(new_x)
    return output


#if input point is inside unit circle
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


    
    """
    t = dict()
    i=0
    for val in itertools.product('10', repeat=2):
        t[val]=XOR((int(val[0]),int(val[1])))
        i+=1
    pretty_print(t)
    """
    """
    t=dict()
    for val in itertools.product('10', repeat=2):
        t[val]= XOR((int(val[0]), int(val[1])))
    pretty_print(t)
    """
    """
    for x in np.arange(-20, 20):
        for y in np.arange(-20, 20):
            ms=2
            if x*0.1 in [0.0, 1.0] and y*0.1 in [0.0,1.0]: ms=5
            if perceptron(step, (1.0,1.0), -0.5, (x,y)) is 1: plt.plot([x*0.1], [y*0.1],'bo', markersize=ms)
            else: plt.plot([x*0.1], [y*0.1], 'ko', markersize=ms)
    plt.show()
        """
    """
    for i in range(int(math.pow(2,4))):
        n= binarycon(i, 4)
        f= plt.figure(i+1)
        w,b= train(i, 2)
        if int(check(n,w,b,2)) is not 1: continue
        for x in np.arange(-20, 20):
            for y in np.arange(-20, 20):
                ms=2
                if x*0.1 in [0.0, 1.0] and y*0.1 in [0.0,1.0]: ms=5
                if perceptron(step, w, b, (x,y)) is 1: plt.plot([x*0.1], [y*0.1],'bo', markersize=ms)
                else: plt.plot([x*0.1], [y*0.1], 'ko', markersize=ms)
    plt.show()
        """
    

    
    """
    for i in range(int(math.pow(2,4))):
        if train(i, 2)==1: count+=1
    print(count)
    for i in range(-2,2):
        for i in range(-2,2):

    count=0
    for i in range(int(math.pow(2,8))):
        if train(i, 3): count+=1
    print(count)
    count=0
    for i in range(int(math.pow(2,16))):
        if train(i, 4): count+=1
    print(count)
"""
