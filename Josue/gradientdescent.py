# gradient descent for linear regression
# using sum of square errors to adjust weights
# follow algorithm from https://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html
import numpy as np
from random import random
import matplotlib
import matplotlib.pyplot as plt

steps = 500
r = .01 # learning rate

x = np.arange(0,10)
# y = np.arange(11,21)
y = np.random.rand(x.shape[0])

a = random()
b = random()

print a
print b


def sse(original, predicted):
    return (np.sum(original - predicted) ** 2) / 2

first = True
for i in range(steps):
    predicted = (x * a) + b
    prevSSE = sse(y, predicted)
    minus = -(y - predicted)
    yyp = np.sum(minus)
    yypx = np.sum(minus * x)

    newa = a - r * yyp
    newb = b - r * yypx

    newPredicted = (x * newa) + newb
    newSSE = sse(y, newPredicted)

    print 'prev: ' + str(prevSSE)
    print 'new: ' + str(newSSE)

    # check direction and change
    if newSSE < prevSSE:
        a = newa
        b = newb
    else:
        a = a + r * yyp
        b = b + r * yypx

print a
print b


# show graphs
fig, ax = plt.subplots()
ax.plot(x, y, 'o')
ax.plot(x, a*x+b, 'k')
plt.show()
