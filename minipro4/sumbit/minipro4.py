# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:42:44 2018

@author: xuyuan
"""

import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from numpy import random

def processData(filename):
    ID = []
    sex = []
    population = []
    nucleobases = []
    file = open(filename,'r').readlines()
    for line in file:
        line = line.split(' ')
        ID.append(line[0])
        sex.append(line[1])
        population.append(line[2])
        line = line[3:-1]
        nucleobases.append(line)
    nucleobases = np.array(nucleobases)
    return ID,sex,population,nucleobases

def processNucleobases(nucleobases):
    most_fre_nuc = []
    for i in range(nucleobases.shape[1]):
        temp = nucleobases[:,i].tolist()
        nuc = Counter(temp)
        most_nuc,most_cnt = nuc.most_common(1)[0]
        most_fre_nuc.append(most_nuc)
    m,n = nucleobases.shape
    binary_matrix = np.zeros((m,n),dtype=int)
    for i in range(nucleobases.shape[1]):
        most_nuc = most_fre_nuc[i]
        for j in range(nucleobases.shape[0]):
            if nucleobases[j][i] != most_fre_nuc[i]:
                binary_matrix[j][i] = 1 
            else:
                binary_matrix[j][i] = 0
    return binary_matrix

#part1 a b
'''
ID,sex,population,nucleobase = processData("p4dataset2018.txt")
binary_matrix = processNucleobases(nucleobase)
print(binary_matrix.shape[0])
print(binary_matrix.shape[1])
'''
#get the different kinds number 
'''
population_num = set(population)
print(len(population_num))
print(population_num)
'''
def part1_b(binary_matrix,population,demesion):
    pca = PCA(demesion)
    nucleobase_pca = pca.fit_transform(binary_matrix)
    pkinds=['ASW','YRI','ACB','ESN','GWD','LWK','MSL']
    kinds = {'ASW':'red','YRI':'black','ACB':'gold','ESN':'green','GWD':'pink','LWK':'blue','MSL':'purple'}
    population_belong = {}
    for i in range(len(population)):
        population_belong[i] = population[i]
    fig, ax = plt.subplots()
    i = 0
    for individual in nucleobase_pca:
        popul = population[i]
        col = kinds[popul]   
        ax.scatter(individual[0], individual[1], c = col)
        i+=1
        
    ax.legend()
    ax.set_title("ASW:red,YRI:black,ACB:gold,ESN:green,GWD:pink,LWK:blue,MSL:purple")
    ax.set_xlabel('v1')
    ax.set_ylabel('v2')
    plt.savefig("part1_b.png",format='png')
    plt.show()
    plt.close()

#part1_b(binary_matrix,population,2)

def part1_d(binary_matrix,population,demesion):
    pca = PCA(demesion)
    nucleobase_pca = pca.fit_transform(binary_matrix)
    pkinds=['ASW','YRI','ACB','ESN','GWD','LWK','MSL']
    kinds = {'ASW':'red','YRI':'black','ACB':'gold','ESN':'green','GWD':'pink','LWK':'blue','MSL':'purple'}
    population_belong = {}
    for i in range(len(population)):
        population_belong[i] = population[i]
    fig, ax = plt.subplots()
    i = 0
    for individual in nucleobase_pca:
        popul = population[i]
        col = kinds[popul]   
        ax.scatter(individual[0], individual[2], c = col)
        i+=1
        
    ax.legend()
    ax.set_title("ASW:red,YRI:black,ACB:gold,ESN:green,GWD:pink,LWK:blue,MSL:purple")
    ax.set_xlabel('v1')
    ax.set_ylabel('v3')
    plt.savefig("part1_d.png",format='png')
    plt.show()
    plt.close()

#part1_d(binary_matrix,population,3)

def part1_f(binary_matrix,demesion):
    pca = PCA(demesion)
    nucleobase_pca = pca.fit_transform(binary_matrix)
    #print(nucleobase_pca.shape)
    #components_ ： 返回模型的各个特征向量。
    vector3=pca.components_[2]
    #print(len(vector[]))
    
    fig, ax = plt.subplots()
    for i in range(10101):
        ax.scatter(i,np.abs(vector3[i]),c='blue')
    ax.legend()
    ax.set_xlabel('nucleobase index')
    ax.set_ylabel('absolute V3')
    plt.savefig("part1_f.png",format='png')
    plt.show()
    plt.close()
    
    
#part1_f(binary_matrix,3)

def pca_recover(X,Y):
    XY=[X,Y]
    XY=np.array(XY)
    XY=XY.T
    #print(XY.shape)
    pca=PCA(2)
    pca.fit(XY)
    #print(len(pca.components_[0]))
    return pca.components_[0][1]/pca.components_[0][0]

def ls_recover(X,Y):
    n=np.dot(X-np.mean(X),Y-np.mean(Y))
    d=((X-np.mean(X))**2).sum()
    return n/d

#part2_a
'''
X_part2a=[x*0.001 for x in range(1,1001)]
Y_part2a=[2*x for x in X_part2a]
print("pca validation:",pca_recover(X_part2a,Y_part2a))
print("ls validation:",ls_recover(X_part2a,Y_part2a))
'''
#part2_b
def part2_b():
    X=np.random.uniform(size=2)
    Y=np.random.uniform(size=2)
    m=X[:]
    n=Y[:]
    recover_pca=pca_recover(m,n)
    recover_ls=ls_recover(X,Y)
    print("X:",X)
    print("Y:",Y)
    print("pca recover:",recover_pca)
    print("ls recover:",recover_ls)

#part2_b()

def part2_c_XY(c):
    X=[xx*0.001 for xx in range(1,1001)]
    X=np.array(X)
    
    Y=[2*X[i] for i in range(0,1000)]
    Y=np.array(Y)
    
    Xnoise=random.randn(1000)*(math.sqrt(c))
    Ynoise=random.randn(1000)*(math.sqrt(c))
    
    Xnoise+=X
    Ynoise+=Y
    
    return X,Xnoise,Ynoise

def part2_c():
    c=[cc*0.05 for cc in range(0,11)]
    times=30
    for i in range(times):
        for j in c:
            X,Xnoise,Ynoise=part2_c_XY(j)
            '''
            if j==0.05:
                print(Xnoise-X)
                break
            '''
            recover_pca=pca_recover(X,Ynoise)
            recover_ls =ls_recover(X,Ynoise)
            plt.plot(j, recover_pca, 'ro', label = 'pca',alpha=1)
            plt.plot(j, recover_ls, 'bo',label='ls',alpha=1)
    plt.title("part2 c")
    plt.xlabel("c")
    plt.ylabel("slope")
    plt.savefig("part2_c.png",format = 'png')
    plt.close()
    
#part2_c()

def part2_d():
    c=[cc*0.05 for cc in range(0,11)]
    times=30
    for i in range(times):
        for j in c:
            X,Xnoise,Ynoise=part2_c_XY(j)
            recover_pca=pca_recover(Xnoise,Ynoise)
            recover_ls =ls_recover(Xnoise,Ynoise)
            plt.plot(j, recover_pca, 'ro', label = 'pca',alpha=1)
            plt.plot(j, recover_ls, 'bo',label='ls',alpha=1)
    plt.title("part2 d")
    plt.xlabel("c")
    plt.ylabel("slope")
    plt.savefig("part2_d.png",format = 'png')
    plt.close()
part2_d()

            
    
 
    
    



    

    
    
    