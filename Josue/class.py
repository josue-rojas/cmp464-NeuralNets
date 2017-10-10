'''
Created on Oct 3, 2017

@author: bob
'''


import numpy as np
from scipy.optimize import minimize# this is so can use their minimize
X=np.array([0,0,1,0,0,1,1,1]).reshape(4,2)
ORY = np.array([0,1,1,1]).reshape(4,1)
ANDY = np.array([0,0,0,1]).reshape(4,1)
XORY = np.array([0,1,1,0]).reshape(4,1)
#===============================================================================
# The above is the basic inputs for truth tables
# that we can use for XOR OR AND ETC
# There are four different inputs of a pair of (0,1)
# that we put in a 4X2 array. This setup will be used in much of
# our future work with neural nets.
# We will make a one layer net here where there will have to be two
# weights and one bias.
#===============================================================================
W= np.random.randn(2,1)#right way for matrix mult
#===============================================================================
# W is the weight matrix which is 2X1 so that we can multiply
# np.dot(X,W) and then add the bias to get an output,
# We dont really have to declare W here as the scipy minimize
# routine is going to use a function called cost and in the
# minimize routine you give a list of the initial weights and the bias
# and the function and the minimize calls the function to figure out a
# minimum from your starting point. We will comment about this later.
#===============================================================================

#===============================================================================
# #the following is a cost function to minimize
# #this setup (not necessarily the cost) will work well with tensorflow
# #but doesnt interface well with scipy minimize.
# #scipy minimize expects the function and the three initial values for w0,w1 and b in a "list"
# # that we have in the 2X1 array W and the scalar b.
#===============================================================================
b=1 # I am thinking of this as the initial value of b
Y=ORY# this is output for the OR

# given our inputs in X

def cost(guess, Y=Y):
    b = guess[-1]
    w = np.array(guess[:-1]).reshape(2,1)
    costArray= ((np.dot(X,w)+b)-Y)**2 # taking square here to eventually take sum of squares
    return costArray.sum()


def gateFunction(Y=Y,gateName='OR'):
    res = minimize(cost,[W[0,0],W[1,0],b],method='Nelder-Mead')
    print ("the w0, w1, and bias b you get are \n", res.x[0],res.x[1],res.x[2])
    Wanswer = np.array([res.x[0], res.x[1]])
    banswer = res.x[2]
    Output = np.dot(X,Wanswer) + banswer
    print ('Output for ', gateName, ': ',Output)

gateFunction(Y=ORY, gateName='OR')
gateFunction(Y=ANDY,gateName='AND')
gateFunction(Y=XORY, gateName='XOR')


#===============================================================================
# #### You must now figure out how to make a decision based on these values
# #### for w0,w1, and b. Try making an output function which is just
# ### output= np.dot(X,Wanswer) + banswer but where Wanswer is from res.x[0],res.x[1]
# #### and banswer is res.x[2]
# ### You will have to make a decision based on how big the output is.
# ###Can you make a decision that gives each of three logical functions from
# ### these outputs? This way of making decisions is a little unsatisfying
# ### and I will show you how with some math with the logistics
# ### function we can make the methods more consistent.
# ### Dont forget that this program is only setup for the OR gate and
# #### you should also do XOR, AND (if it will work ??????)
# ### really for each of these gates you need only change the answers
# ### your expect -- that is change value(values???) in the variable Y
# ### Can you use the logistic function to think of a way to
# ### convert your output to a probability to get an answer
#===============================================================================
