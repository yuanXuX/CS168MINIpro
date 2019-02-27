# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:42:24 2018

@author: xuyuan
"""
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
import random
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

train_n = 100
test_n = 1000
d = 100
trials=10


def f_error(X,a,y):
    n=np.matmul(X,a)
    n-=y
    n=math.sqrt(np.sum(np.square(n)))
    d=math.sqrt(np.sum(np.square(y)))
    return float(n)/d

def get_a(X_train,y_train):
    inv = np.linalg.inv(X_train)
    a = np.matmul(inv, y_train) 
    return a

def part_2a(func,choice,c):
    train_err=0
    test_err=0
    
    for i in range(trials):
        X_train = np.random.normal(0,1, size=(train_n,d))
        a_true = np.random.normal(0,1, size=(d,1))
        y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
        X_test = np.random.normal(0,1, size=(test_n,d))
        y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))
        if choice=='a':
            a=func(X_train,y_train)
        else:
            a=func(X_train,y_train,c)
        tra=f_error(X_train,a,y_train)
        tes=f_error(X_test,a,y_test)
        train_err+=tra
        test_err+=tes
    train_err=float(train_err)/trials
    test_err=float(test_err)/trials
    if choice=='a':
        print("part a train error:",train_err)
        print("part a test error:",test_err)
    else:
        print("part b train error:",train_err)
        print("part b test error:",test_err)
        return train_err,test_err
        

#part_2a(get_a,'a',c="")

######################################
c=[0.0005,0.005,0.05,0.5,5,50,500]

def get_a2(X, y,C):
    res=np.matmul(X.T, X) + C * np.identity(d)
    res=np.linalg.inv(res)
    a = np.matmul(res, X.T)
    a = np.matmul(a, y)
    return a

    
def part_2b():
    error_tra=[]
    error_tes=[]
    for i in c:
        tra,tes=part_2a(get_a2,'b',i)
        error_tra.append(tra)
        error_tes.append(tes)
    plt.xlabel('lamda')
    plt.ylabel('average error')
    plt.plot(c,error_tra,color='red',label='train error',marker='o')
    plt.plot(c,error_tes,color='green',label='test error',marker='o')
    plt.legend() # 显示图例
    plt.show()
    plt.savefig("part2_b.png", format = 'png')
    plt.close()


'''
part_2b()
'''
################################################################################
#part2_c
steps=[0.00005,0.0005,0.005]
steps_train_error=[]
steps_test_error=[]
iterations=1000000
trials=10

def part_2c():
    for step in steps:
        train_error=0
        test_error=0
        for trial in range(trials):
            X_train = np.random.normal(0,1, size=(train_n,d))
            a_true = np.random.normal(0,1, size=(d,1))
            y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
            X_test = np.random.normal(0,1, size=(test_n,d))
            y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))      
            a=np.zeros(shape=(d,1))
            for i in range(iterations):
                random_i=np.random.randint(0,train_n-1)
                train=X_train[random_i].reshape((d,1))
                gradient=2*train*(a.T.dot(train)-y_train[random_i])
                a-=step*gradient
            tra=f_error(X_train,a,y_train)
            tes=f_error(X_test,a,y_test)
            train_error+=tra
            test_error+=tes
        train_error/=10
        test_error/=10
        steps_train_error.append(train_error)
        steps_test_error.append(test_error)
    plt.xlabel('steps')
    plt.ylabel('average error')
    plt.plot(steps,steps_train_error,color='red',label='train error',marker='o')
    plt.plot(steps,steps_test_error,color='green',label='test error',marker='o')
    plt.legend() # 显示图例
    plt.show()
    plt.savefig("part2_c.png", format = 'png')
    plt.close()
'''
part_2c()
'''

####################################################################################################

steps=[0.00005,0.005]
train_errors={}

test_errors={}

l2_errors={}


iterations=1000000
X_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))      
 
xx=[i for i in range(1,iterations+1)]
xx100=[ i*100 for i in range(1,int(iterations/100)+1)]
           
def part_2d():
    
    for step in steps:
        a=np.zeros(shape=(d,1))
        for i in range(1,iterations+1):
            random_i=np.random.randint(0,train_n-1)
            train=X_train[random_i].reshape((d,1))
            gradient=2*train*(a.T.dot(train)-y_train[random_i])
            a-=step*gradient
            error=f_error(X_train,a,y_train)
            train_errors.setdefault(step,[]).append(error)
            
            if i%100==0:
                error=f_error(X_test,a,y_test)
                test_errors.setdefault(step,[]).append(error)
            
            l2_errors.setdefault(step,[]).append(np.linalg.norm(a))
    part2_d_plot1()
    part2_d_plot2()
    part2_d_plot3()
    

def part2_d_plot1():            
    plt.xlabel('iterations')
    plt.ylabel('normalized training error')
    plt.plot(xx,train_errors[steps[0]],color='red',label='step=0.00005')
    plt.plot(xx,train_errors[steps[1]],color='green',label='step=0.005')
    fa=[f_error(X_train,a_true,y_train) for i in range(iterations)]
    plt.plot(xx,fa,color='blue',label='a true error')
    plt.legend() # 显示图例
    plt.show()
    plt.savefig("part2_d1.png", format = 'png')
    plt.close()
def part2_d_plot2():            
    plt.xlabel('iterations')
    plt.ylabel('normalized training error')
    plt.plot(xx100,test_errors[steps[0]],color='red',label='step=0.00005')
    plt.plot(xx100,test_errors[steps[1]],color='green',label='step=0.005')
    plt.legend() # 显示图例
    plt.show()
    plt.savefig("part2_d2.png", format = 'png')
    plt.close()
def part2_d_plot3():            
    plt.xlabel('iterations')
    plt.ylabel('norm of SGD')
    plt.plot(xx,l2_errors[steps[0]],color='red',label='step=0.00005')
    plt.plot(xx,l2_errors[steps[1]],color='green',label='step=0.005')
    plt.legend() # 显示图例
    plt.show()
    plt.savefig("part2_d3.png", format = 'png')
    plt.close()
'''
part_2d()
'''
#########################################################################
iterations=1000000
step=0.00005
rs=[0,0.1,0.5,1,10,20,30]
train_errors=[]
test_errors=[]

def part_2e():
    for r in rs:
        a=np.ones(shape=(d,1))*r
        train_e=0
        test_e=0
        for i in range(iterations):
            random_i=np.random.randint(0,train_n-1)
            train=X_train[random_i].reshape((d,1))
            gradient=2*train*(a.T.dot(train)-y_train[random_i])
            a-=step*gradient
            error=f_error(X_train,a,y_train)
            train_e+=error
            error=f_error(X_test,a,y_test)
            test_e+=error
        train_errors.append(float(train_e)/iterations)
        test_errors.append(float(test_e)/iterations)
    plt.xlabel('radius')
    plt.ylabel('average error with 1000000 iterations')
    plt.plot(rs,train_errors,color='red',label='train error')
    plt.plot(rs,test_errors,color='green',label='test error')
    plt.legend() # 显示图例
    plt.show()
    plt.savefig("part2_e.png", format = 'png')
    plt.close()

part_2e()
      

    
            
            
            
            
        

            
        
        

        
        

    
    
    

        

    