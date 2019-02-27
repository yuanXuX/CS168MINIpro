# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 09:38:56 2018

@author: xuyuan
"""
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
import random
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#PART 1
d=100
n=1000
X=np.random.normal(0,1,size=(n,d))
a_true=np.random.normal(0,1,size=(d,1))
y=X.dot(a_true)+np.random.normal(0,0.5,size=(n,1))

def get_a(flag):
    if flag:
        return np.zeros(shape=(d,1))
    Xt=X.T
    XtX=np.matmul(Xt,X)
    XtXinv=np.linalg.inv(XtX)
    XtXinvXt=np.matmul(XtXinv,Xt)
    a=np.matmul(XtXinvXt,y)#a:d*1
    return a

def get_error(a):  
    sqrd_error=0
    at=a.T# at:1*d
    for i in range(n):
        xi=X[i].reshape((d,1))
        sqrd_error+=np.power(at.dot(xi)-y[i],2)
    return sqrd_error

def part_a():
    flag=0
    a=get_a(flag)
    error=get_error(a)
    flag=1
    aa=get_a(flag)
    errorr=get_error(aa)
    print("normal error:",error)
    print("set a all 0's error:",errorr)

#part_a
part_a()

#part_b
learning_rates=[0.00005, 0.0005, 0.0007]
iterations=20

'''
y=(y_predict-y_true)^2
y'=2*y_predict-y_true
'''
def gradient_descent():
    gdresult = {}
    for i in learning_rates:
        a=np.zeros(shape=(d,1))
        for j in range(iterations):
            gd=0
            for elem in range(n):
                xi=X[elem].reshape((d,1))
                gd+=2*xi*(a.T.dot(xi)-y[elem])
            a-=i*gd
            loss=get_error(a)
            gdresult.setdefault(i,[]).extend(loss)
    return gdresult

def make_plot(gdresult,iterations,outFileName):
    xlabs=[i for i in range(1,iterations+1)]
    plt.xlabel('iteration times')
    plt.ylabel('loss')    
    plt.plot(xlabs,gdresult[learning_rates[0]],color='green',label='lr=0.0005')
    plt.plot(xlabs,gdresult[learning_rates[1]],color='red',label='lr=0.005')
    plt.plot(xlabs,gdresult[learning_rates[2]],color='blue',label='lr=0.01')
    plt.legend() # 显示图例
    plt.show()
    plt.savefig(outFileName, format = 'png')
'''
gdresult=gradient_descent()
make_plot(gdresult,iterations,"part_b.png")
print("0.00005:",gdresult[learning_rates[0]][-1])
print("0.0005:",gdresult[learning_rates[1]][-1])
print("0.0007:",gdresult[learning_rates[2]][-1])
'''

#part_c
learning_rates=[0.0005,0.005, 0.01]
iterations=1000
def stochastic_gradient_descent():
    gdresult = {}
    for i in learning_rates:
        a=np.zeros(shape=(d,1))
        for j in range(iterations):
            random_choose=np.random.randint(0,n-1)
            xi=X[random_choose].reshape((d,1))
            gd=2*xi*(a.T.dot(xi)-y[random_choose])
            a-=i*gd
            loss=get_error(a)
            gdresult.setdefault(i,[]).extend(loss)
    return gdresult
            
gdresult=stochastic_gradient_descent()
make_plot(gdresult,iterations,"part_c.png")
print("0.0005:",gdresult[learning_rates[0]][-1])
print("0.005:",gdresult[learning_rates[1]][-1])
print("0.01:",gdresult[learning_rates[2]][-1])

        
            
    
   

    
    