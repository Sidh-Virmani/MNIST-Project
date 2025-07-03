from utilities import dense, output_layer

def sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = dense(x,  W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = output_layer(a2, W3, b3)
    return(a3)