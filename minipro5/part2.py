# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:26:18 2018

@author: xuyuan
"""

from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from collections import Counter

def k_appro(k,U,S,V):
    UU=U[:,:k]
    print("U shape:",U.shape)
    
    SS=np.diag(S)
    SS=SS[:k,:k]
    print("S shape:",S.shape)
    
    VV=V[:k,:]
    print("VT shape:",VV)
    
    new_picture=np.matmul(UU,SS)
    new_picture=np.matmul(new_picture,VV)
    return new_picture
    
def part2():
    picture=io.imread("p5_image.gif",flatten=True)
    #print(picture)
    U,S,V=np.linalg.svd(picture)
    print("Total numbers of singular value:",len(S))
    #print(U)
    
    k=[1,3,10,20,50,100,150,200,400,800,1170]
    for i in k:
        new_picture=k_appro(i,U,S,V)
        plt.imshow(new_picture,cmap="gray")
        plt.savefig("rank_"+str(i)+".png",format='png')
    
part2()
        

