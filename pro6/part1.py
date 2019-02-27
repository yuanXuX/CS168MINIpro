# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:49:59 2018

@author: xuyuan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections as cs

n=100

def part1_b_a():
    line_D=np.zeros((n,n))
    for i in range(n):
        if i==0 or i==n-1:
            line_D[i][i]=1
        else:
            line_D[i][i]=2
    eig_value,eig_vector=np.linalg.eig(line_D)
    part1_plot(eig_value,eig_vector,"part1_b_a_D","part1_b_a_D")
    line_A=np.zeros((n,n))
    for i in range(n-1):
        line_A[i][i+1]=1
        line_A[i+1][i]=1
    print(line_A)
    line_L=line_D-line_A
    eig_value,eig_vector=np.linalg.eig(line_A)
    part1_plot(eig_value,eig_vector,"part1_b_a_A","part1_b_a_A")
    return line_L

def part1_b_b():
    line_D=np.zeros((n,n))
    for i in range(n):
        if i==0 or i==n-2:
            line_D[i][i]=2
        elif i==n-1:
            line_D[i][i]=n-1
        else:
            line_D[i][i]=3
    eig_value,eig_vector=np.linalg.eig(line_D)
    part1_plot(eig_value,eig_vector,"part1_b_b_D","part1_b_b_D")
    line_A=np.zeros((n,n))
    for i in range(n-1):
        line_A[i][i+1]=1
        line_A[i+1][i]=1
        line_A[n-1][i]=1
        line_A[i][n-1]=1
    line_L=line_D-line_A
    eig_value,eig_vector=np.linalg.eig(line_A)
    part1_plot(eig_value,eig_vector,"part1_b_b_A","part1_b_b_A")
    return line_L

def part1_b_c():
    line_D=np.zeros((n,n))
    for i in range(n):
            line_D[i][i]=2
    eig_value,eig_vector=np.linalg.eig(line_D)
    part1_plot(eig_value,eig_vector,"part1_b_c_D","part1_b_c_D")
    line_A=np.zeros((n,n))
    for i in range(n-1):
        line_A[i][i+1]=1
        line_A[i+1][i]=1
    line_A[n-1][0]=1
    line_A[0][n-1]=1
    line_L=line_D-line_A
    eig_value,eig_vector=np.linalg.eig(line_A)
    part1_plot(eig_value,eig_vector,"part1_b_c_A","part1_b_c_A")
    return line_L

def part1_b_d():
    line_D=np.zeros((n,n))
    for i in range(n):
        if i==n-1:
            line_D[i][i]=n-1
        else:
            line_D[i][i]=3
    eig_value,eig_vector=np.linalg.eig(line_D)
    part1_plot(eig_value,eig_vector,"part1_b_d_D","part1_b_d_D")
    line_A=np.zeros((n,n))
    for i in range(n-1):
        line_A[i][i+1]=1
        line_A[i+1][i]=1
        line_A[n-1][i]=1
        line_A[i][n-1]=1
    line_A[0][n-2]=1
    line_A[n-2][0]=1
    line_L=line_D-line_A
    eig_value,eig_vector=np.linalg.eig(line_A)
    part1_plot(eig_value,eig_vector,"part1_b_d_A","part1_b_d_A")
    return line_L

def part1_plot(eig_value,eig_vector,title,filename):
    index_and_value=dict(enumerate(eig_value))
    counter=cs.Counter(index_and_value)
    sorted_value=counter.most_common()
    largest_value=sorted_value[0][0]
    second_largest_value=sorted_value[1][0]
    smallest_value=sorted_value[-1][0]
    second_smallest_value=sorted_value[-2][0]   
    x=[i for i in range(n)]
    plt.scatter(x,eig_vector[:,largest_value],color='red',marker='o',label="largest")
    plt.scatter(x,eig_vector[:,second_largest_value],color='yellowgreen',marker='o',label="second_largest")
    plt.scatter(x,eig_vector[:,smallest_value],color='purple',marker='o',label="smallest")
    plt.scatter(x,eig_vector[:,second_smallest_value],color='dodgerblue',marker='o',label="second_smallest")
    plt.title(title)
    plt.xlabel("i")
    plt.ylabel("vector")
    plt.legend(shadow=True, loc = 0)
    plt.savefig(filename + ".png", format = 'png')
    plt.close()
    
def part1_b():
    part1_b_a()
    part1_b_b()
    part1_b_c()
    part1_b_d()

#part1_b()

def part1_plot_c(vec1,vec2,title,filename):
    plt.plot(vec1,vec2,'-o')
    plt.title(title)
    plt.xlabel("v2")
    plt.ylabel("v3")
    plt.savefig(filename + ".png", format = 'png')
    plt.close()
    
def part1_c():
    total=[]
    total.append(part1_b_a())
    total.append(part1_b_b())
    total.append(part1_b_c())
    total.append(part1_b_d())
    filename=["part1_c_a","part1_c_b","part1_c_c","part1_c_d"]
    for i in range(4):
        current=total[i]
        eig_value,eig_vector=np.linalg.eig(current)
        index_and_value=dict(enumerate(eig_value))
        counter=cs.Counter(index_and_value)
        sorted_value=counter.most_common()
        second=eig_vector[:,sorted_value[-2][0]]
        third=eig_vector[:,sorted_value[-3][0]]
        file=filename[i]
        title=filename[i]
        part1_plot_c(second,third,title,file)

#part1_c()

dn=500
def dist(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def part1_plot_d(x,y,vec1,vec2,title,filename):
    plt.plot(vec1,vec2,'bo',label='v2-v3')
    flag=0
    for i in range(dn):
        if x[i]<0.5 and y[i]<0.5 and flag==0:
            plt.plot(vec1[i],vec2[i],'ro',label='original',alpha=0.3)
            flag=1
        if x[i]<0.5 and y[i]<0.5 and flag==1:
            plt.plot(vec1[i],vec2[i],'ro')
    plt.title(title)
    plt.xlabel("v2")
    plt.ylabel("v3")
    plt.legend(shadow=True, loc = 0)
    plt.savefig(filename + ".png", format = 'png')
    plt.close()
    
def part1_d():
    x=np.random.uniform(0,1,dn)
    y=np.random.uniform(0,1,dn)
    A=np.zeros((dn,dn))
    D=np.zeros((dn,dn))
    for i in range(dn):
        for j in range(dn):
            if j==i:
                continue
            elif dist(x[i],x[j],y[i],y[j])<=0.25:
                A[i][j]=1
                A[j][i]=1
        total=0
        for k in range(dn):
            if A[i][k]==1:
                total+=1
        D[i][i]=total
    L=D-A
    eig_value,eig_vector=np.linalg.eig(L)
    index_and_value=dict(enumerate(eig_value))
    counter=cs.Counter(index_and_value)
    sorted_value=counter.most_common()
    second=eig_vector[:,sorted_value[-2][0]]
    third=eig_vector[:,sorted_value[-3][0]]
    title="part1_d"
    file=title
    part1_plot_d(x,y,second,third,title,file)
    

part1_d()
        
        
        
        
    
        
        
