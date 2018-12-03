# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 12:23:53 2018

@author: xuyuan
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import collections as cs
import pandas as pd
import networkx as nx

def getData(filename):
    data=pd.read_csv(filename, header = None) 
    return data.as_matrix()
data=getData("cs168mp6.csv")
n=1495

def part2_b(data):
    A=np.zeros((n,n))
    D=np.zeros((n,n))
    for i in range(len(data)):
        friend1=data[i][0]
        friend2=data[i][1]
        A[friend1-1][friend2-1]=1
        A[friend2-1][friend1-1]=1
    for i in range(n):
        count=0
        for j in range(n):
            if A[i][j]==1:
                count+=1
        D[i][i]=count
    L=D-A
    eig_value,eig_vector=np.linalg.eig(L)
    index_and_value=dict(enumerate(eig_value))
    counter=cs.Counter(index_and_value)
    sorted_value=counter.most_common()
    nums=len(sorted_value)
    for i in range(12):
        print(sorted_value[nums-1-i][1])
    return A,D,L,sorted_value,eig_vector

def part2_plot_c(vec1,vec2,title,filename):
    plt.plot(vec1,vec2,'bo',label='v2-v3')
    plt.title(title)
    plt.xlabel("v2")
    plt.ylabel("v3")
    plt.legend(loc = 0)
    plt.savefig(filename + ".png", format = 'png')
    plt.close()
    
A,D,L,eig_values_12,eig_vector=part2_b(data)

def part2_c(eig_values_12,eig_vector_12):
    second_smallest=eig_vector[:,eig_values_12[-2][0]]
    third_smallest=eig_vector[:,eig_values_12[-3][0]]
    part2_plot_c(second_smallest,third_smallest,"part2_c_analysis","part2_c")
    

#part2_c(eig_values_12,eig_vector_12)
Graph=nx.from_numpy_matrix(A)
#nx.draw(Graph)
def part2_d(L):
    eig_value,eig_vector=np.linalg.eig(L)
    index_and_value=dict(enumerate(eig_value))
    counter=cs.Counter(index_and_value)
    sorted_value=counter.most_common()
    x=[x for x in range(1,n+1)]
    for i in range(5,9):
        y=eig_vector[:,sorted_value[-1*i][0]]
        plt.scatter(x,y)
        plt.title("personID and vector")
        plt.xlabel("person")
        plt.ylabel("vector")
        plt.legend(loc = 0)
        plt.show()
        filename='part2_d_find_'+str(i)
        plt.savefig(filename + ".png", format = 'png')
        plt.close()
    #choose eig_vec_(-7)
    best_vec=eig_vector[:,sorted_value[-8][0]]
    set1=[]
    set2=[]
    set3=[]
    for i in range(n):
        if best_vec[i]<=-0.03 and best_vec[i]>-0.05:
            set1.append(i)
        elif best_vec[i]>=0 and best_vec[i]<0.01:
            set2.append(i)
        elif best_vec[i]>=0.07 and best_vec[i]<0.09:
            set3.append(i)
    set1_=[i for i in range(n) if i not in set1]
    set2_=[i for i in range(n) if i not in set2]
    set3_=[i for i in range(n) if i not in set3]
    cdt1=nx.algorithms.cuts.conductance(Graph,set1,set1_)
    cdt2=nx.algorithms.cuts.conductance(Graph,set2,set2_)
    cdt3=nx.algorithms.cuts.conductance(Graph,set3,set3_)
    print("10 items in set1:")
    print(set1[:10])
    print("the conductance of set1:",cdt1 )
    print("10 items in set2:")
    print(set2[:10])
    print("the conductance of set1:",cdt2 )
    print("10 items in set3:")
    print(set3[:10])
    print("the conductance of set1:",cdt3 )

#part2_d(L)

def part2_e(L):
    random_set=random.sample(range(0,n),150)
    random_set_=[x for x in range(n) if x not in random_set]
    cdt=nx.algorithms.cuts.conductance(Graph,random_set,random_set_)
    print("random result:",cdt)
part2_e(L)
    
    
    




    
    

        
            
        
        
    
    