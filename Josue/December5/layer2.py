
'''Created on Jun 7, 2017

'''
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import display, Image
from scipy import ndimage
import pickle
import tensorflow as tf

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print("how do labels look \n")

for i in range(15,21):
    print(train_labels[i])

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)#takes the Array to 2 dimensional array NoDataXNofeatures
  # https://stackoverflow.com/questions/37867354/in-numpy-what-does-selection-by-none-do
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)# to make ONE HOT ENCODING; the None adds a dimension and tricky numpy broadcasting
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print("after changing into proper format for our training ")
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print("how do labels look \n")
for i in range(15,21):
    print(train_labels[i,:])


# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround. This is use of minibatches
#It is also called stochastic gradient descent; Mahtematicallly decent size minibatches approximate statistics of big sample
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))# predictions will be one hot encoded too and seeing if agree where 1 is
          / predictions.shape[0])
batch_size = 128# the N for the minibatches
other_size = 1000

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))#the input data
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, other_size]))
  biases1 = tf.Variable(tf.zeros([other_size]))
  weights2 = tf.Variable(tf.truncated_normal([other_size, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  y1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  logits = tf.matmul(y1, weights2) + biases2
  loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))


  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits=logits) # the softmax computes the probabilities from outputs by using sigmoid and normalizing
  y1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  valid_logits = tf.matmul(y1_valid, weights2) + biases2
  valid_prediction = tf.nn.softmax(logits=valid_logits)

  y1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
  test_logits = tf.matmul(y1_test, weights2) + biases2
  test_prediction = tf.nn.softmax(logits=test_logits)

num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
