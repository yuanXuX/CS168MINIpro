# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:36:16 2018

@author: xuyuan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#均值函数
def meanX(X):
    #axis=0表示用列来求均值
    #为了减去各自特征的平均值
    return np.mean(X,axis=0)
a=[[1,2,3],[2,3,4]]
y=meanX(a)
print(np.tile(y,(2,1)))

def pca(X,k):
    average=meanX(X)
    m,n=np.shape(X)
    data_adjust=[]
    avge=np.tile(average,(m,1))
    data_adjust=X-avge
    #计算协方差矩阵
    covX=np.cov(data_adjust.T)
    #求解协方差矩阵的特征值和特征向量
    fea_val,fea_vec=np.linalg.eig(covX)
    #依照特征值进行降序排序
    index=np.argsort(-fea_val)
    res=[]
    if k>n:
        return
    else:
        select_vec=np.matrix(fea_vec.T[index[:k]])
        res=data_adjust*select_vec.T
    return res


    
