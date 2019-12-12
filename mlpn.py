import numpy as np
import loglinear as ll
import math

STUDENT={'name': 'ZAIDMAN IGAL',
         'ID': '311758866'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    l = x
    for i in range(len(params) -1):
        W,b = params[i]
        np.dot(l, W)
        l = np.tanh(np.dot(l, W) + b)
        
    return ll.classifier_output(l, params[-1])


def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    U, b_tag = params[-1]
    
    y_vec = np.zeros(U.shape[1])
    y_vec[y] = 1
    gradients = []
    
    y_tag = classifier_output(x, params)
    loss = -math.log(y_tag[y])

    z_s = [x]
    a_s = [x]

    for W, b in params:
        z_s.append(z_s[-1].dot(W) + b)
        a_s.append(np.tanh(z_s[-1]))

    diff = y_tag - y_vec
    gradients.insert(0, [np.array([a_s[-2]]).transpose().dot(np.array([diff])), diff])

    for i in (range(len(params) - 1))[::-1]:
        gb = gradients[0][1].dot(params[i + 1][0].transpose()) * (1 - np.power(a_s[i+1], 2))
        gW = np.array([z_s[i]]).transpose().dot(np.array([gb]))
        gradients.insert(0, [gW, gb])

    return loss, gradients

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(len(dims) - 1):
        W = np.random.randn(dims[i], dims[i + 1]) / np.sqrt(dims[i])
        b = np.zeros(dims[i + 1])
        params.append([W, b])

    return params

