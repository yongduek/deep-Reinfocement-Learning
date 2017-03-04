
# coding: utf-8

# # Tensor Graph for data visualization
# 

# In[1]:

import numpy as np
xy = np.loadtxt ('train-xor.txt', unpack=True)
xdata = xy[0:-1].transpose() # 0 에서 끝-1 까지
ydata = xy[-1:].transpose()
print ('xdata', xdata, '\n', 'ydata', ydata)
print ('xdata[0].shape=', xdata[0].shape, xdata.shape[0], xdata.shape[1])
print ('ydata.shape=', ydata.shape)


# ## Specifying Unknown input batch size!
# 
# - The input data may have unknown number of data. 
# - The codes below shows how to specify such inputs/output data.
# - Notice that they are all defined as 2D tensors

# In[2]:

import tensorflow as tf

X = tf.placeholder (tf.float32, name='Xinput', shape=[None,xdata.shape[1]])#+list(xdata[0].shape))
print ([None] + list(xdata[0].shape))
Y = tf.placeholder (tf.float32, name='Yinput', shape=[None]+list(ydata[0].shape))
print ([None]+list(ydata[0].shape))


# # Now, 2-layer network
# 
# The network has now twoo layers. This will increase the capacity of the network and result in a correctly fitted model.

# In[3]:

nhidd = 12
W1 = tf.Variable (tf.random_uniform([xdata.shape[1],nhidd], -1., 1.), name='W1')
b1 = tf.Variable (tf.zeros([nhidd]), name='b1')
W2 = tf.Variable (tf.random_uniform([nhidd,1],-1.,1.), name='W2')
b2 = tf.Variable (tf.zeros([1]), name='b2')

h1 = tf.sigmoid( tf.matmul (X, W1) + b1)
xor_out = tf.sigmoid (tf.matmul(h1, W2) + b2)

yxlog = Y * tf.log(xor_out) + (1-Y)*tf.log(1.-xor_out)
cost = -tf.reduce_mean ( yxlog )


# In[ ]:

# minimize
opti = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train2 = opti.minimize(cost)


# In[4]:

def evaluate(ss, xor_out, feed_dict):
    pred = ss.run (xor_out, feed_dict={X:xdata, Y:ydata})
    print ('pred=', pred.transpose())
    cpred = ss.run( tf.equal(tf.floor(xor_out+0.5), Y), feed_dict )
    print ('cpred=', cpred.transpose())
    accuracy = tf.reduce_mean (tf.cast(cpred, 'float'))
    print ('accuracy=', accuracy.eval(feed_dict))


# In[ ]:

with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())
        
    # data fitting
    # 
    for i in range(10000):
        ss.run (train2, feed_dict={X:xdata, Y:ydata})
        if i%1000==0:
            print ('------- data fitting evaluation ------')
            print ('iteraiton: ', i, 
                  ' cost= ', ss.run(cost, feed_dict={X:xdata, Y:ydata}))
            evaluate (ss, xor_out, feed_dict)            
            
    # model test
    #
    evaluate (ss, xor_out, feed_dict)


# ## EOF
