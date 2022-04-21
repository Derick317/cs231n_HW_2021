from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    C = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        f=[]
        for j in range(C):
            f.append(W[:,j] @ X[i])
        f -= max(f)
        expf=np.exp(f)
        summation = np.sum(expf)
        ratio = expf[y[i]] / summation
        loss -= np.log(ratio)
        dratio = - 1.0 / ratio
        dsum = - dratio * expf[y[i]] / (summation ** 2)
        dexpf = []
        for j in range(C):
            dexpf.append(dsum)
        dexpf[y[i]] += dratio / summation
        df = np.array(expf) * np.array(dexpf)
        for j in range(C):
            dW[:, j] += df[j] * X[i].T
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    f = X @ W # shape: (num_train, num_class)
    f = (f.T - np.max(f, axis = 1)).T
    expf = np.exp(f) # shape: (num_train, num_class)
    summation = np.sum(expf, axis = 1) # shape: num_train
    correct_exp = expf[range(num_train), y] # shape: num_train
    ratio = correct_exp / summation # shape: num_train
    logarithm = np.log(ratio)  # shape: num_train
    loss -= np.sum(logarithm) 
    
    #Backprop
    dratio = - 1 / ratio # shape: num_train
    dsum = - dratio * correct_exp / (summation * summation) # shape: num_train
    dexpT = np.zeros(f.shape).T + dsum # shape: (num_class, num_train)
    dexpf = dexpT.T # shape: (num_train, num_class)
    dexpf[range(num_train), y] += dratio / summation # shape: (num_train, num_class)
    df = dexpf * expf
    dW = X.T @ df

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
