'''
Created on Oct 29, 2017

@author: bob
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #to avoid warnings about compilation
import tensorflow as tf
log_dir=os.path.dirname(os.path.realpath(__file__))
os.system('rm event*') #assuming names are like that?!?!?!1
arf= tf.constant([3,3],name="arf")
barf= tf.constant([[1,5],[3,6]],name="barf")
print("shape of array \n",barf.shape,"\n")
a= tf.add(arf,barf,name="addarfbarf")
#a=3+5 #try this to show wont work
b= a+7

with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_dir,sess.graph )
    aa,bb =sess.run([a,b])
    print("sum \n",aa,"\n\n","add another \n", bb)
