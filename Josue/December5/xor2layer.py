import tensorflow as tf


# Input and out
X = tf.constant([0,0,0,1,1,0,1,1], shape=[4,2], dtype=tf.float32)
Y = tf.constant([0,1,1,0], shape=[4,1], dtype=tf.float32)

learning_rate = 0.01
steps = 100000

weights1 = tf.Variable(tf.random_uniform([2,2]), name="weights1")
weights2 = tf.Variable(tf.random_uniform([2,1]), name="weights2")
# these should also be random
bias1 = tf.Variable([0,0], name="bias1", dtype=tf.float32)
bias2 = tf.Variable([0], name="bias2", dtype=tf.float32)

# layer 1 and 2
layer1 = tf.sigmoid(tf.matmul(X, weights1) + bias1)
layer2 = tf.sigmoid(tf.matmul(layer1, weights2) + bias2)

# cost function is the average
cost = tf.reduce_mean(( (Y * tf.log(layer2)) +
        ((1 - Y) * tf.log(1.0 - layer2)) ) * -1)

minimized = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# check if valid
ExpectedBool = [False, True, True, False]
def checkValid(results):
    j = 0
    for result in results:
        if (result[0] > .5) != ExpectedBool[j]:
            return False
        j+=1
    return True

for i in range(steps):
    sess.run(minimized)
    # check if answer is gotten so wont keep going
    # probably should check every so step cause this has a for loop
    if checkValid(sess.run(layer2)):
        break



# print the resuls
print('Results out:', [x[0] > .5 for x in sess.run(layer2)])
print('weights for layer 1 ', sess.run(weights1))
print('bias 1 ', sess.run(bias1))
print('weights for layer 2 ', sess.run(weights2))
print('bias 2 ', sess.run(bias2))
