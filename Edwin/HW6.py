
import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input data-set (each row is training, columns are input nodes)
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],                                                                                 
                [1,1,1] ])
    
# output data-set (1 row , 4 columns)           
y = np.array([[0,0,1,1]]).T  # Transposed array                                                                                        

# seed random numbers to make calculation
# deterministic 
np.random.seed(1)

# initialize weights randomly with mean 0
W0 = 2*np.random.random((3,4)) - 1 # Weights
W1 = 2*np.random.random((4,1)) - 1 # Weights

for j in range(60000):

    # forward propagation(full batch)layers 0, 1, and 2
    l0 = X
    # hidden layers(multiplies then passes output through sigmoid)/the guess
    l1 = nonlin(np.dot(l0,W0))
    l2 = nonlin(np.dot(l1,W1))
    
    # how much did we miss?
    l2_error = y - l2 # element wise subtraction

    # (error weighted derivative)multiply 'element-wise' how much it missed by the 
    # slope of the sigmoid at the values in l2
    l2_delta = l2_error * nonlin(l2,deriv=True)# nonlin() generates the slopes/reducing the error of high confidence predictions

# how much did each l1 value contribute to the l2 error (according to the weights)?[back propagation!]
    l1_error = l2_delta.dot(W1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    # update weights
    W1 += l1.T.dot(l2_delta)
    W0 += l0.T.dot(l1_delta)

print ("Output After Training:")
print (l2)
