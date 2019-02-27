# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:01:20 2018

@author: xuyuan
"""

import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
import random

train_n = 100
test_n = 10000
d = 200
trials = 1000

#from part2
def get_a(X, y,C):
    res=np.matmul(X.T, X) + C * np.identity(d)
    res=np.linalg.inv(res)
    a = np.matmul(res, X.T)
    a = np.matmul(a, y)
    return a
#from part 2
def f_error(X,a,y):
    n=np.matmul(X,a)
    n-=y
    n=math.sqrt(np.sum(np.square(n)))
    d=math.sqrt(np.sum(np.square(y)))
    return float(n)/d

def part3():
    traerror = 0.0
    teserror = 0.0
    c_for_train = 0.0005
    c_for_test=0.05
    for trial in range(trials):
        X_train = np.random.normal(0,1, size=(train_n,d))
        a_true = np.random.normal(0,1, size=(d,1))
        y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
        X_test = np.random.normal(0,1, size=(test_n,d))
        y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))
        a = get_a(X_train,y_train,c_for_train)
        tra= f_error(X_train, a, y_train)
        a = get_a(X_test,y_test,c_for_test)
        tes= f_error(X_test, a, y_test)
        traerror += tra
        teserror += tes
    ave_train_err=traerror/trials
    ave_test_err=teserror/trials
    print("my result")
    print("average train error:",ave_train_err)
    print("average test error:",ave_test_err)

part3()



