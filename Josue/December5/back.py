import tensorflow as tf
from random import random

# X = [[0, 0],[0, 1],[1, 0],[1, 1]]
# Y = [[0],[1],[1],[0]]
# Input and output
X = tf.constant([0,0,0,1,1,0,1,1], shape=[4,2], dtype=tf.float32)
Y = tf.constant([0,1,1,0], shape=[4,1], dtype=tf.float32)


x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

weights1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="weights1")
weights2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="weights2")

bias1 = tf.Variable(tf.zeros([2]), name="bias1")
bias2 = tf.Variable(tf.zeros([1]), name="bias2")

A2 = tf.sigmoid(tf.matmul(x_, weights1) + bias1)
Hypothesis = tf.sigmoid(tf.matmul(A2, weights2) + bias2)

cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) +
        ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100000):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
    if i % 1000 == 0:
        print('Epoch ', i)
        print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
        print('weights1 ', sess.run(weights1))
        print('bias1 ', sess.run(bias1))
        print('weights2 ', sess.run(weights2))
        print('bias2 ', sess.run(bias2))
        print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
