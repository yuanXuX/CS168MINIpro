# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:13:23 2018

@author: xuyuan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from collections import Counter
filecsv="co_occur.csv"
filetxt="dictionary.txt"

def getData(filecsv):
    co_occurence=pd.read_csv(filecsv,header=None)
    co_occurence=np.array(co_occurence)
    co_occurence=np.log(co_occurence+1)
    return co_occurence

def saveData(filename,data,isMatrix):
    file=open(filename,'w')
    if isMatrix:
        for i in data:
            for j in i:
                towrite=str(j)
                file.write(towrite+' ')
            file.write('\n')
    else:
        for i in data:
            towrite=str(i)
            file.write(towrite+' ')
        file.write('\n')
    file.close()


#part1 1b
def part1_b():
    co_occurence,words=getData(filecsv,filetxt)
    print(co_occurence.shape)
    U,singular_value,VT=randomized_svd(co_occurence,n_components=100,random_state=None)
    #saveUSV to save time
    saveData("U.txt",U,True)
    saveData("SV.txt",singular_value,False)
    saveData("VT.txt",VT,True)
            
    #plot
    x=[i for i  in range(100)]
    y=singular_value
    plt.title("singular values of M")
    plt.xlabel("x")
    plt.ylabel("singular value")
    plt.plot(x,y,'o')
    plt.savefig("part1_b.png", format = 'png')
    plt.close()

#part1_b()

def part1_c():
    dictionary=open('dictionary.txt','r').readlines()
    words=[]
    for line in dictionary:
        line=line.split()
        words.append(line[0])
    
    file=open("U.txt",'r').readlines()
    U=[]
    for line in file:
        line=line.split(' ')
        array=[]
        for num in line:
            if num!='\n':
                num=float(num)
                array.append(num)
        U.append(array)
    U=np.array(U)
    print(U.shape)
    
    for i in range(5):
        singular_value=U.T[i]
        index_and_value=dict(enumerate(singular_value))
        sv_counter=Counter(index_and_value)
        sv_sorted=sv_counter.most_common()
        big_val=sv_sorted[0:11]
        small_val=sv_sorted[-10:]
        print("biggest in :",i)
        for index,value in big_val:
            print(words[index]," ",value)
        print("smallest in :",i)
        for index,value in small_val:
            print(words[index]," ",value)
part1_c()
    
    
            
            
        
        
        
    



        
    
    




