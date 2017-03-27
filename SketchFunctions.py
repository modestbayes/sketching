
# coding: utf-8

# In[2]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# In[2]:

# best linear unbiased estimate
def lm(X, Y):
    estimate = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), Y))
    return(estimate)


# In[3]:

def get_ratio(beta_sketch, beta_true):
    return(np.mean(np.square(beta_sketch - np.transpose(beta_true))) / np.linalg.norm(beta_true))


# In[4]:

def generateData(n, beta, link='linear', sigma=1.0):
    """Generate data for GLM.
    
    # Arguments
        N: number of observations
        beta: regression coefficient vector
        link: linear or logistic
        sigma: standard deviation of Gaussian noise (only for linear regression)
        
    # Returns
        A list of design matrix and response vector
    """
    d = beta.shape[0]
    X = np.random.normal(0, 1, n * d).reshape(n, d)
    eta = np.dot(X, beta)
    if link == 'linear':
        mu = eta
        Y = eta + np.random.normal(0, sigma, n).reshape(n, 1)
    elif link == 'logistic':
        print('here')
        mu = 1.0 / (1.0 + np.exp(-eta))
        Y = (mu > 0.5).astype(float)
    return(X, Y)


# In[5]:

# hessian (or use tf.hessians in Tensorflow 1.0)
def compute_hessian(beta, grad, D):
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


# In[6]:

def hadamard(k):
    """Create standard Hadamard matrix.
    
    # Arguments
        k: power of 2
    
    # Returns
        A Hadamard matrix of size 2 ^ k
    """
    H2 = np.ones((2, 2))
    H2[1, 1] = -1.0
    H2 = H2 / np.sqrt(2)
    H = 1.0
    for i in range(0, k):
        H = np.kron(H2, H)
    return(H)


# In[7]:

def sketch(X, Y, gamma, method='gaussian'):
    """Randomly project data.
    
    # Arguments
        X: design matrix
        Y: response vector
        gamma: project constant >
            R = gamma * D: projection dimension
        
    # Usage
        Subsampling: method='none', sampling=True, bootstrap=True
        i.i.d Gaussian: method='gaussian', sampling=False, bootstrap=False
        Hadamard: method='hadamard', sampling=True, bootstrap=False
    
    # Returns
        A list of design matrix and response vector in projection space
    """
    N = X.shape[0]
    D = X.shape[1]
    R = gamma * D
    SX = X
    SY = Y
    if method == 'gaussian':
        sampling = False
        bootstrap = False
        S = np.random.normal(0, 1, R * N).reshape(R, N)
        SX = np.dot(S, X)
        SY = np.dot(S, Y)
    elif method == 'hadamard':
        sampling = True
        bootstrap = False
        H = hadamard(int(np.log2(N)))
        temp = np.ones(N)
        temp[np.random.randint(low=0, high=N, size=int(N / 2))] = -1.0
        D = np.diag(temp)
        S = np.dot(H, D)
        SX = np.dot(S, X)
        SY = np.dot(S, Y)
    elif method == 'sampling':
        sampling = True
        bootstrap = True
    if sampling:
        select = np.random.choice(np.array(range(0, N)), size=R, replace=bootstrap)
        SX = SX[select, :]
        SY = SY[select, :]
    return(SX, SY)

# In[8]:

def simulate(N, D, link = 'linear', sigma = 1.0,
             gamma = 4, method = 'gaussian', 
             MAXITER = 50, TOL = 10**(-9),
             path_out = 'path.csv'):

    print("*********************************\n")
    print("Beginning IHS Simulation")
    print("-----------------------------\n")
    print("    N = ", N)
    print("    D = ", D)
    print("    link = ", link)
    print("    gamma = ", gamma)
    print("    method = ", method)
    print("------------------------------\n")
    ### Generate Data, set Non-TF variables ###
    beta_true = np.random.uniform(-2, 2, D).reshape(D, 1)
    dataX, dataY = generateData(N, beta_true, link=link, sigma=sigma)
    path = np.zeros((MAXITER, D + 1))
    ###
    
    ### Set up TensorFlow Computation Graph ###
    X = tf.placeholder('float', [None, D]) 
    Y = tf.placeholder('float', [None, 1])
    beta = tf.Variable(tf.random_normal([D, 1], stddev=1.0))
    if link == 'linear':
        # linear regression with mean squared error
        Y_hat = tf.matmul(X, beta)
        loss = tf.reduce_sum(tf.square(Y - Y_hat))
    elif link == 'logistic':
        # logistic regression with log loss
        eta = tf.matmul(X, beta)
        p = 1.0 / (1.0 + tf.exp(-eta))
        loss = -1.0 * (tf.reduce_sum(Y * tf.log(p) + (1.0 - Y) * tf.log(1 - p)))    
        
    # Gradient of loss function
    grad = tf.gradients(loss, beta)

    # Hessian and Fisher Information
    hessian = compute_hessian(beta, grad, D)
    fisher = tf.matrix_inverse(hessian)

    # Update beta by delta
    delta = tf.placeholder('float', [D, 1])
    update = beta.assign_add(delta)
    ###
    
    ### Run Simulation ###
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        stepsize = 1.0
        print("Iterate: ")
        for i in range(MAXITER):
            if i % 10 == 0:
                print(i//10 + 1, end='.')

            # Sketching with chosen method and compression factor
            sketchX, sketchY = sketch(dataX, dataY, gamma, method='gaussian')

            # Compute Gradient
            g = sess.run(grad, feed_dict={X: sketchX, Y: sketchY})[0]

            # Compute Fisher Info
            I = sess.run(fisher, feed_dict={X: sketchX, Y: sketchY})

            # Take Newton-Raphson Step
            stepsize = 1.0
            current_loss = sess.run(loss, feed_dict={X: dataX, Y: dataY})
            sess.run(update, feed_dict={delta : -np.dot(I, g) * stepsize})
            new_loss = sess.run(loss, feed_dict={X: dataX, Y: dataY})

            # Backstepping to ensure decrease in loss function
            while (new_loss > current_loss) and (stepsize > TOL):
                stepsize = stepsize / 2.0
                sess.run(update, feed_dict={delta : np.dot(I, g) * stepsize})
                new_loss = sess.run(loss, feed_dict={X: dataX, Y: dataY})
            stepsize = min(1.0, 2 * stepsize)

            # Note step and new error ratio
            path[i, :-1] = np.transpose(beta.eval())
            path[i, -1] = get_ratio(path[i, :-1], beta_true)
    print("\n------------------------------\n")
    ###
    np.savetxt(path_out, path)
    
    print("Error Ratio: ", path[-1, -1], "\n")
    print("*** Simulation Complete ***")
    return(path)


# In[9]:

def plot_path(path, coeff_ind=[0, 1]):
    MAXITER = np.shape(path)[0]
    print(MAXITER)
    step = range(MAXITER)
    ratio = path[:, -1]

    # Error Ratio Plot
    plt.figure(1)
    plt.plot(step, ratio)
    plt.title("Error Ratio")

    # Plots for beta paths
    plt.figure(2)

    plt.subplot(221)
    plt.plot(path[:, coeff_ind[0]], path[:, coeff_ind[1]])
    plt.plot(path[MAXITER - 1, coeff_ind[0]], path[MAXITER - 1, coeff_ind[1]], 'rx')
    plt.title("Beta x Beta Path")

    plt.subplot(222)
    plt.plot(step, path[:, coeff_ind[0]])
    plt.title("Beta Path")

    plt.show()