'''
Created on Oct 3, 2017

@author: hector
'''

import numpy as np
from pylab import meshgrid
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import stats

print("\n####### THIS IS OUR DATA #######")
X = np.array([0,0,1,0,0,1,1,1])
X = np.reshape(X, (4,2))
print("X values:\n",X,"\nShape of X: ",X.shape)


print("\n####### THIS IS OUR Y #######")
Y = np.array([0,1,1,0]).reshape(4,1)
print("Y Values:\n",Y,"\nShape of Y: ",Y.shape)

print("\n####### THIS IS OUR BIAS #######")
# b=1
b= np.arange(2,4).reshape(1,2)
print("BIAS =", b)



print("\n####### THIS IS OUR W #######")
W=np.random.randn(2,2)##right wat for matrix multiplication
print("Wvalues:\n",W,"\nShape of W: ",W.shape)


print("\n####### THIS IS X*W without a Function #######")
outInt = np.dot(X,W)
print("Outint:\n",outInt)


##this does the X*W+b-Y but in a function!
#Wv is a variable, bv is a variable
def cost(Wv,bv):
    costArray=((np.dot(X,Wv)+bv)-Y)**2 #X and Y are not variables is the original data
    return costArray.sum()
    

#     costArray=(((Wv*X)+bv)-Y)**2 #X and Y are not variables is the original data

def costWrapper(Wwrongv):
    Wwrongv= Wwrongv.tolist()
    b1 = Wwrongv.pop()
    Wright1 = np.array(Wwrongv).reshape(2,1)
    return cost(Wright1, b1)
    

print("\n####### THIS IS OUR Funtion Result #######")
outIntv = cost(W,b)
print("Outint:\n",outIntv)


res = minimize(costWrapper, [W[0,0],W[1,0],b[0,1]]  ,method='Nelder-Mead')
print("result is ",res)

# (res.x[0],res.x[1]) =>  X * W   rexres.x[2] = B
output =np.dot(X,[res.x[0],res.x[1]])+res.x[2]
print("Or output  /n ")
print(output)



