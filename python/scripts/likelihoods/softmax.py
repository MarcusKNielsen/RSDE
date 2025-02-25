import numpy as np

def softmax(x,A):
    P = np.exp(A@x)
    Z = np.sum(P)
    P = P / Z
    return P