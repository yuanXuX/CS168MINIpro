# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:10:51 2018

@author: xuyuan
"""
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from PIL import Image

#part1 a
file="wonderland-tree.txt"
data=open(file,'r').readlines()
n=1200

def part1_a():
    num_of_1=0
    total=0
    #注意最后一个换行符
    for row in data:
        for num in row[:-1]:
            if num=='1':
                num_of_1+=1
            total+=1
    print("k/n:",num_of_1/total)
part1_a()

'''
reference code:
    
    import cvxpy as cp
    import numpy as np
    
    # Problem data.
    m = 30
    n = 20
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    
    # Construct the problem.
    x = cp.Variable(n)
    
    objective = cp.Minimize(cp.sum_squares(A*x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)
    
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    
    # The optimal value for x is stored in `x.value`.
    print(x.value)
    
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    print(constraints[0].dual_value)
'''
def part1_b(r=600):
    A=np.random.randn(n,n)
    xarray=[]
    for row in data:
        for item in row[:-1]:
            xarray.append(int(item))
    #600*1200
    xarray=np.array(xarray)
    #print(len(b))
    b=np.dot(A[:r],xarray)
    x=cp.Variable(n)
    objective=cp.Minimize(cp.norm(x, 1))
    constraints=[b==A[0:r]*x,x>=0,x<=1]
    prob=cp.Problem(objective,constraints)
    result=prob.solve(cp.ECOS_BB)
    #
    print("Validation:",np.allclose(x.value,xarray))
    
#part1_b()

#part3
def get_x_value(r):
    A=np.random.randn(n,n)
    xarray=[]
    for row in data:
        for item in row[:-1]:
            xarray.append(int(item))
    xarray=np.array(xarray)
    b=np.dot(A[:r],xarray)
    x=cp.Variable(n)
    objective=cp.Minimize(cp.norm(x, 1))
    constraints=[b==A[0:r]*x,x>=0,x<=1]
    prob=cp.Problem(objective,constraints)
    result=prob.solve(cp.ECOS_BB)
    return x.value

def part1_c():
    mini=.001
    xarray=[]
    for row in data:
        for item in row[:-1]:
            xarray.append(int(item))
    xarray=np.array(xarray)
    similar=[]
    #use binary search
    begin=1
    end=n-1
    while begin<end:
        middle=begin+int((end-begin)/2)+1
        result=np.linalg.norm(get_x_value(middle)-xarray,1)
        if result<mini:
            similar.append(middle)
            end=middle
        else:
            if begin==middle:
                break
            begin=middle
    find=min(similar)
    print("min r is:",find)
#part1_c()

getR=460
#part d
def part1_d(r=460):
    x=[r+i for i in range(-10,3)]
    x_value=[]
    for i in x:
        x_value.append(np.linalg.norm(get_x_value(i),1))
    plt.plot(x,x_value)
    plt.title("part1_d")
    plt.xlabel("i")
    plt.ylabel("norm(xi-x)")
    plt.savefig("part1_d.png",format="png")
    plt.show()
#part1_d()
    
def part2_a():
    # Load the images.
    orig_img = Image.open("stanford-tree.png")
    corr_img = Image.open("corrupted.png")
    
    # Convert to arrays.
    Uorig = np.array(orig_img)[:,:,0]
    Ucorr = np.array(corr_img)[:,:,0]
    rows,cols=Uorig.shape
    # Known is 1 if the pixel is known,
    # 0 if the pixel was corrupted.
    Known = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
             if Uorig[i, j] == Ucorr[i, j]:
                Known[i, j] = 1
    return Uorig,Ucorr,Known
    
def part2_b():
    Uorig,Ucorr,Known=part2_a()
    m,n=Ucorr.shape
    result=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if Known[i][j]=='1':
                result[i][j]=int(Ucorr[i][j])
            else:
                neighbors=[]
                if i-1>=0:
                    neighbors.append(int(Ucorr[i-1][j]))
                if i+1<=m-1:
                    neighbors.append(int(Ucorr[i+1][j]))
                if j-1>=0:
                    neighbors.append(int(Ucorr[i][j-1]))
                if j+1<=n-1:
                    neighbors.append(int(Ucorr[i][j+1]))
                ave=sum(neighbors)/len(neighbors)
                result[i][j]=ave           
    plt.imshow(result)
    plt.savefig("part2_b.png", format = 'png')
part2_b()

def part2_c():
    Uorig,Ucorr,Known=part2_a()
    U=cp.Variable(Ucorr.shape)  
    obj=cp.Minimize(cp.tv(U))   
    constraints = [cp.multiply(Known, U) == cp.multiply(Known, Ucorr)]   
    prob=cp.Problem(obj, constraints)   
    prob.solve(verbose = True)
    plt.imshow(U.value)  
    plt.savefig("part2_c.png", format = "png")
part2_c()
    


    
    






                