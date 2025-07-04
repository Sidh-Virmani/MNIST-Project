import numpy as np

def relu(z):
    return np.maximum(0, z)

def dense(a_in, W, b):        #computes the activation of dense layers of the network 
    
    a_out = relu(np.dot(a_in, W) + b)
    return a_out

def output_layer(a_in, W, b):
    output = np.dot(a_in, W) + b
    return output

def softmax(z):
    ez = np.exp(z)
    sum_ez = np.sum(ez)
    z_new = ez/sum_ez
    return z_new

def sparse_categorical_cross_entropy(y_hat, y_label):  #yhat is softmax output, y_label is the true value
    return -np.log(y_hat[y_label] + 1e-8)  # 1e-8 prevents log(0)

# if probability y_hat very close to 1 thn loss almost 0, if probability very close to 0, loss tends to infinite