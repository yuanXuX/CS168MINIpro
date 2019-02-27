# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:51:07 2018

@author: xuyuan
"""

import numpy as np
a=[[1,2],[5,6],[11,12]]
s,v,d=np.linalg.svd(a)
print(s.shape)
print(v.shape)
print(d.shape)

a=np.matmul(s,v)

print(a)