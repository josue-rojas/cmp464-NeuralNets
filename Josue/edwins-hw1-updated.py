'''
Created on Sep 12, 2017

@author: EDWIN
this is an update version just made the perceptron Algorithm into a function to not repeat code
'''
import numpy as np
from random import choice
import math
from pylab import ylim
import matplotlib.pyplot as plt

#after running the program several times it is difficult to determine which is faster as the run time is very fast on my machine.
#XOR never resolves

unit_step = lambda x: 0 if x < 0 else 1

training_data_or = [
    (np.array([0,0,1]), 0),
    (np.array([0,1,1]), 1),
    (np.array([1,0,1]), 1),
    (np.array([1,1,1]), 1), ]

training_data_xor = [
    (np.array([0,0,1]), 0),
    (np.array([0,1,1]), 1),
    (np.array([1,0,1]), 1),
    (np.array([1,1,1]), 0), ]

training_data_and = [
    (np.array([0,0,1]), 0),
    (np.array([0,1,1]), 0),
    (np.array([1,0,1]), 0),
    (np.array([1,1,1]), 1), ]


def percAlgo(training_data, gateName=''):
    w = np.random.rand(3)
    print('\nStarting seed',w)
    errors = []
    errors2 = []
    bias = 0.2
    steps = 100

    print('Perceptron Algorithm '+ gateName)

    for _ in range(steps):
        x, expected = choice(training_data)
        result = np.dot(w, x)
        error = expected - unit_step(result)
        errors.append(error)
        w += bias * error * x

    for x, _ in training_data:
        result = np.dot(x, w)
        print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

    ylim([-1,1])
    plt.plot(errors)
    plt.show()

percAlgo(training_data_or, 'OR')
percAlgo(training_data_and, 'AND')
percAlgo(training_data_xor, 'XOR')
