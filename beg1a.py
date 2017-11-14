'''
Created on Oct 29, 2017

@author: bob
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #to avoid warnings about compilation
import tensorflow as tf
log_dir=os.path.dirname(os.path.realpath(__file__))
os.system('rm event*') #assuming names are like that?!?!?!1
counter=3
a= tf.add(counter,5)
#a=3+5 #try this to show wont work
b= a+7
#merged=tf.summary.scalar("our output ",a)
tf.summary.scalar("aa",a)
tf.summary.scalar("bb",b)
merged = tf.summary.merge_all()
#sess=tf.Session()
with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_dir,sess.graph )
    for i in range(6):
        sumab,aa,bb =sess.run([merged,a,b])
        print(aa,bb)
        counter=counter+1
        writer.add_summary(sumab,i)
    #writer.flush()  # seems not needed if there is a loop but be careful
    #could use writer.close() also I think
