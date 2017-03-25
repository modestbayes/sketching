
# coding: utf-8

# In[766]:

import numpy as np
import tensorflow as tf
import time
import random
from math import log
from tensorflow.python.client import timeline


# In[767]:

random.seed(1234)
# N observations with D parameters
N = 2**11
D = 1000
# projection dimension
gamma = 4
m = gamma*D
# variance parameter
sigma = 1
# Y = Xb + (0, sigma)
beta_true = np.random.uniform(-10, 10, D).reshape(D, 1)
#beta_true = np.array([[1], [2], [-3], [4], [5]])
dataX = np.random.normal(0, 1, N * D).reshape(N, D)
dataY = np.random.normal(0, sigma, N).reshape(N, 1) + np.matmul(dataX, beta_true)


# In[768]:

beta_LS = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(dataX), dataX)), np.transpose(dataX)), dataY)


# In[769]:

# placeholding tensors and variable
X = tf.placeholder('float', [None, D]) 
Y = tf.placeholder('float', [None, 1.0])
beta = tf.Variable(tf.random_normal([D, 1], stddev=1.0))


# In[770]:

# linear regression with mean squared error
Y_hat = tf.matmul(X, beta)
MSE = tf.add(tf.reduce_sum(tf.square(Y - Y_hat)), tf.norm(beta, ord = 1))
#MSE = tf.reduce_sum(tf.square(Y - tf.matmul(X, beta)))


# In[771]:

# gradient
grad = tf.gradients(MSE, beta)


# In[772]:

# hessian (or use tf.hessians in Tensorflow 1.0)
def compute_hessian():
    for i in range(D):
        # element in the gradient vector
        dfdx_i = tf.slice(grad[0], begin=[i, 0], size=[1, 1])
        # differentiate again
        ddfdx2_i = tf.gradients(dfdx_i, beta)[0]
        # combine second derivative vectors
        if i == 0:
            hess = ddfdx2_i
        else:
            hess = tf.concat([hess, ddfdx2_i], 1)
    return(hess)

hessian = compute_hessian()


# In[773]:

# fisher information
fisher = tf.matrix_inverse(hessian)


# In[774]:

# update beta by delta
delta = tf.placeholder('float', [D, 1])
drop = beta.assign_add(delta)


# In[775]:

beta_sketch = np.zeros((D, 1))

for MAXITER in [30]:
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range(0, MAXITER):
            # gaussian random projection
            S = np.random.normal(0, 1, m * N).reshape(m, N)
            SX = np.dot(S, dataX)
            SY = np.dot(S, dataY)
            # compute gradient
            g = sess.run(grad, feed_dict={X: SX, Y: SY})[0]
            # compute hessian
            I = sess.run(fisher, feed_dict={X: SX, Y: SY},  options=run_options, run_metadata=run_metadata)
            # drop
            sess.run(drop, feed_dict={delta : -np.dot(I, g)})
        beta_sketch = beta.eval()
        ratio = np.linalg.norm(beta_sketch - beta_LS) / np.linalg.norm(beta_LS)
        print(m, N, MAXITER, ratio)
        
        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
print(np.linalg.norm(beta_LS))


# In[776]:

#ratio = np.mean(np.square(beta_true - beta_sketch)) / np.mean(np.square(beta_true - beta_LS))
#ratio = np.linalg.norm(beta_sketch - beta_LS) / np.linalg.norm(beta_LS)

#print(m, N, MAXITER, ratio, np.linalg.norm(beta_LS))
#print(np.linalg.norm(beta_LS - beta_true))
#print(np.linalg.norm(beta_sketch - beta_true))
#print(np.linalg.norm(beta_LS - beta_sketch))
#print(K/N)


# In[777]:

#print(beta_LS, '\n\n', beta_sketch)

