import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #to avoid warnings about compilation
import tensorflow as tf
log_dir=os.path.dirname(os.path.realpath(__file__))
os.system('cd' + log_dir + '; rm event*')

# Multiply  4 matrices of dimensions (16,8),(8,4),(4,2),(2,1)
# (16,8)
first = tf.constant([x for x in range(128)],name="sixteen-eight", shape=[16,8])
second = tf.constant([x for x in range(32)],name="eight-four", shape=[8,4])
third = tf.constant([x for x in range(8)],name="four-two", shape=[4,2])
fourth = tf.constant([x for x in range(2)],name="two-one", shape=[2,1])
#
fM = tf.matmul(first, second)
fMlogistic = 1 / (1 + np.exp(math.e ** (-1*((np.dot(X,w)+b)-Y))))
sM = tf.matmul(fM, third)
tM = tf.matmul(sM, fourth)


# other way to do it one line
# matMul = tf.matmul(tf.matmul(tf.matmul(tf.constant([x for x in range(128)],name="sisteen", shape=[16,8]), tf.constant([x for x in range(32)],name="eight", shape=[8,4])), tf.constant([x for x in range(8)],name="four", shape=[4,2])), tf.constant([x for x in range(2)],name="two", shape=[2,1]))

with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_dir,sess.graph)
    # answer = sess.run([matMul]) #other way graph
    answer = sess.run([fM, sM, tM])
    print answer
