import numpy as np

def relu(z):
    return np.maximum(0, z)

def dense(a_in, W, b):        #computes the activation of dense layers of the network 
    
    a_out = relu(np.dot(a_in, W) + b)
    return a_out

def output_layer(a_in, W, b):
    output = np.dot(a_in, W) + b
