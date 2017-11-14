'''
Created on Oct 29, 2017

@author: bob
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #to avoid warnings about compilation
import tensorflow as tf
log_dir=os.path.dirname(os.path.realpath(__file__))
os.system('rm event*') #assuming names are like that?!?!?!

a= tf.add(3,5)
#a=3+5 #try this to show wont work
b= a+7
c= b+6

with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_dir,sess.graph )
    aa,bb,cc =sess.run([a,b,c])
    print(aa,bb,cc)
