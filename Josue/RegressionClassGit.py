
'''
Created on Sep 1, 2017
@author: bob

Homework and Classwork Next time(9/12) -- We will have a long lab to see how you are doing
HW&LAB(easy) Change the cost function look at graph for program from class of 9/5
HW&LAB(hard if need to graph) Change the input layer in 9/5 program  to have two features for each x ( N inputs where each input  point is in two dimensions)
should be easy to use minimize in higher dimensions
hard to graph as in 4 dimensions-- could try scatter plots and color to indicate cost.
How about 3 or more features. Try without graphing. Is there any way to visualize?

'''
import numpy as np
from pylab import meshgrid
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import stats

# constants
costType = 'squared' #type of cost used 'absolute', 'squared', 'cubed'
rng = np.random
M= 4 # should be None or an int
N= 3 # number of different points for regression; I am only allowing values of -1 and 1
Xarray=10*rng.randn(N) if not M else 10*rng.randn(N, M)
print("Xvalues ",Xarray,"\nshape of Xarray ",Xarray.shape)
Vlow=-10
Vhigh=10 #originally 0 and 2 to mimic perceptron
V= rng.randint(size=(N,M), low=Vlow, high=Vhigh)
V=2*V-1  #note that the constant 2 is broadcast through the array as it should in linear algebra
Weight = rng.randint(size=(N), low=Vlow, high=Vhigh)
print("values taken on by V",V, "\nshape of V \n",V.shape)


# gets the slope and intercept?(i think) (SHOULD CHANGE TO CHECK FOR MULTIPLE REGRESSION)
if not M and N==2 and Xarray[1]!= Xarray[0] :
    wans= (V[1]-V[0])/(Xarray[1]-Xarray[0])
    bans= V[0]-wans*Xarray[0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(Xarray,V)

def cost(w,b,cost=costType):
    if cost == 'squared':
        return (((np.dot(Xarray,w)+b)-V)**2).sum() # taking square here and then take the sum
    elif cost=='absolute':
        return np.absolute(((np.dot(Xarray,w)+b)-V)).sum() # taking the absolute value and the sum of all
    elif cost=='cubed':
        return (((np.dot(Xarray,w)+b)-V)**2).sum() # taking the vubed value and the sum of all
#print ("cost = \n",cost(1,1))

def costWrapper(Avar):# just a wrapper function for sciPy
    return cost(Avar[0],Avar[1]) # I think I can give it an ordered pair
def cost1(a,b):# to test the plot
    return a*b

#
#vcostWrapper= np.vectorize(costWrapper)  #did not work wanted to be able to work on array of pairs

#
#print("usingwrapper ", costWrapper([1,1]))
#find the minimum from sciPy ; sort of ridiculous for square as is normal regression you can solve analytically
res =  minimize(costWrapper,[list(Weight)],method='Nelder-Mead')# this method doesnt use derivatives

print("result is ",res)
print("the minimum it reached is ",res.fun)
if not M and N== 2 and Xarray[1]!= Xarray[0] :
    print("the actual answer which should give us 0 as a cost is \n ",wans,bans)
    print("\n the answer from regression slope, intercept,error is \n",slope,intercept,cost(slope,intercept))
# lets checkout how w and b grow
incr=1
#===============================================================================
# print("growth of w and b around mininimizer inxcrements of \n", incr)
# for i in range(0,10,incr):
#     print("cost from w increments ", cost(res.x[0]+i,res.x[1]))
#     print("cost from b increments ", cost(res.x[0],res.x[1]+i))
#===============================================================================



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# !!!!!!!!!!!!!!!!! should change -.8 to 1 to really see more correct graph !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
w= np.arange(res.x[0]-2,res.x[0]+2,.05)  #change from 1 to -.8 to see arrays
b= np.arange(res.x[1]-2,res.x[1]+2,.05)
#w= np.arange(-10,10,.05)  #change from 1 to -.8 to see arrays
#b= np.arange(-10,10,.05)
W,B= meshgrid(w,b) #grid of points
print("\n W array\n",W,"\n and shape of W \n",W.shape)
print("\nB array\n",B,"\n and shape of B \n",B.shape)
#===============================================================================
# Ztemp= list(zip(np.ravel(W),np.ravel(B)))
# Ztemp= np.array(Ztemp)
# print ("just zipped arrays as list of pairs \n",Ztemp, "\n and length of list \n",len(Ztemp))
# ZtempA= vcostWrapper(Ztemp,1)
#
#
# ZtempA= ZtempA.reshape(W.shape)
# print("\n reshaped array of z values \n",ZtempA)
#===============================================================================

zs=np.array([cost(w,b)for w,b in zip(np.ravel(W),np.ravel(B))]) #note is iterator
Z = zs.reshape(W.shape)
print(" \n  the Z after applying function \n",Z,"\n and shape of Z \n",Z.shape)

ax.plot_surface(W, B, Z)

ax.set_xlabel('W Label')
ax.set_ylabel('B Label')
ax.set_zlabel('Z Label')

plt.show()
