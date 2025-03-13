import numpy as np


def likelihood(x1,x2,y=None,p=()):
    
    A = np.array([[1, -1], [0.0, -0.0], [-1, 1]])
    
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
    
        X1,X2 = np.meshgrid(x1,x2)
        X_stack = np.stack([X1, X2], axis=-1)
        Ax = np.einsum('kj, nmj -> knm', A, X_stack)
        P = np.exp(Ax)
        Z = np.sum(P, axis=0)
        P = P / Z
        
    else:
        
        x = np.array([x1,x2])
        P = np.exp(A@x)
        Z = np.sum(P)
        P = P / Z

    if y == None:
        return P
    else:
        return P[1-y]
