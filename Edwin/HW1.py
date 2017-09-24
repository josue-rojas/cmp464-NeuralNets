'''
Created on Sep 12, 2017

@author: EDWIN
'''
import numpy as np
from random import choice 
from numpy import array, dot, random
import operator
import math
from datetime import datetime
from pylab import ylim 
import matplotlib.pyplot as plt

start_time = datetime.now()

# Sum the squares of the first 20 odd numbers.
def oddSqrSum(n):
    return n*(2*n+1)*(2*n-1)/3
print (oddSqrSum(20))


print(" ")#space for new problem

# a dictionary that returns the name of the tallest person
d = {'Kathy': 1.70, 'Joan': 1.65, 'Nancy': 1.60}
def tall():
    tallest = max(d, key=d.get)
    return tallest
print(tall())

print(" ")#space for new problem

#  A list of the names in the dictionary sorted by height.
def dictsort():
    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    return sorted_d
print (dictsort())

print(" ")#space for new problem

#The distance between two vectors
rng = np.random      
N = 10
vector2=10*rng.rand(N)
vector1 = [0.7539606023871248, 0.21901116926068442, 0.8238483697504183, 0.7778771476736663, 0.5486707627827041, 0.48157988814108477, 0.45790607965732033, 0.6313090112329858, 0.8256065064430328, 0.2074759568343847]

v1 = np.array(vector1)
v2 = np.array(vector2)

def euclidean1 (vector1, vector2):
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    def millis():
        dt = datetime.now() - start_time
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        return ms
    print("Array Distance Time",millis())
#     print(" ")#space for new problem
    return dist

def euclidean2(vector1, vector2):
    #use numpy.dot() function
    diff = vector2 - vector1
    squareDistance = np.dot(diff.T, diff)
    def millis():
        dt = datetime.now() - start_time
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        return ms
    print("Dot Product Distance Time",millis())
#     print(" ")#space for new problem
    return math.sqrt(squareDistance)

print('Array distance =', euclidean1(vector1,vector2))
print('Dot product distance =', euclidean2(vector1, vector2))

#after running the program several times it is difficult to determine which is faster as the run time is very fast on my machine.

#Perceptron Algorithm OR/XOR
#by changing the training set you can clearly see that XOR never resolves
print(" ")#space for new problem
print('Perceptron Algorithm OR/XOR')

unit_step = lambda x: 0 if x < 0 else 1 

training_data_or = [ 
    (array([0,0,1]), 0), 
    (array([0,1,1]), 1), 
    (array([1,0,1]), 1), 
    (array([1,1,1]), 1), ]

training_data_xor = [ 
    (array([0,0,1]), 0), 
    (array([0,1,1]), 1), 
    (array([1,0,1]), 1), 
    (array([1,1,1]), 0), ]  

w = random.rand(3)
print('\nStarting seed',w)
errors = []
errors2 = [] 
bias = 0.2 
steps = 100 

for _ in range(steps): 
    x, expected = choice(training_data_or) #change set to see XOR 
    result = dot(w, x) 
    error = expected - unit_step(result) 
    errors.append(error) 
    w += bias * error * x 
    
for x, _ in training_data_or: #change to xor to see results of XOR
    result = dot(x, w) 
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

#plot errors to see learning
ylim([-1,1]) 
plt.plot(errors)
plt.show()
