'''
Created on Oct 29, 2017

@author: bob
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #to avoid warnings about compilation
import numpy as np
import tensorflow as tf
log_dir=os.path.dirname(os.path.realpath(__file__))
os.system('cd' + log_dir + '; rm event*') #assuming names are like that?!?!?!1
arfInitial= np.arange(2,dtype=np.int32).reshape((2,1))## very tricky need to match data types with numpy and tensorflow
print("this will be initial value of arf \n",arfInitial)
arf= tf.Variable(arfInitial,name="arf")

#arfA=tf.reshape(arf,[2,1])# cant change tensor shape while programming
print("shape of array arf \n",arf)
#print("\nvalue of array arf\n",arf) #cant print nicely is a tensor
barf= tf.Variable([[1,5],[3,6]],name="barf")
print("shape of array barf\n",barf.shape,"\n")
#print("\n value of array barf \n",barf)# cant print easily is tensor
a= tf.add(arf,barf,name="addarfbarf")
#a=3+5 #try this to show wont work
b= a+7
#arf=arf+7  #takes away tensor property of arf dont do

arfNew = tf.matmul(barf,arf)
arfAssign= arf.assign(arfNew)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_dir,sess.graph )
    sess.run(init)
    aa,bb,arfCalc,pr =sess.run([a,b,arf,arfAssign]) #order of arfCalc and pr dont seem to matter
    print("sum \n",aa,"\n\n","add another \n", bb)
    print("\n new value of barf*arf \n",pr)
    print("\n value of arf \n",arfCalc)
    aa,bb,pr =sess.run([a,b,arfAssign])
    print("sum \n",aa,"\n\n","add another \n", bb)
    print("\n new value of barf*arf \n",pr)
