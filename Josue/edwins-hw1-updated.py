'''
Created on Sep 12, 2017

@author: EDWIN
this is an update version just made the perceptron Algorithm into a function to not repeat code
'''
from random import choice
from numpy import array, dot, random
import math
from pylab import ylim
import matplotlib.pyplot as plt

#after running the program several times it is difficult to determine which is faster as the run time is very fast on my machine.
#XOR never resolves

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

training_data_and = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 0),
    (array([1,0,1]), 0),
    (array([1,1,1]), 1), ]


def percAlgo(training_data, gateName=''):
    w = random.rand(3)
    print('\nStarting seed',w)
    errors = []
    errors2 = []
    bias = 0.2
    steps = 100

    print('Perceptron Algorithm '+ gateName)

    for _ in range(steps):
        x, expected = choice(training_data)
        result = dot(w, x)
        error = expected - unit_step(result)
        errors.append(error)
        w += bias * error * x

    for x, _ in training_data:
        result = dot(x, w)
        print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

    ylim([-1,1])
    plt.title(gateName)
    plt.plot(errors)
    plt.show()

percAlgo(training_data_or, 'OR')
percAlgo(training_data_and, 'AND')
percAlgo(training_data_xor, 'XOR')
