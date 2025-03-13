import numpy as np
from scipy.special import binom


def likelihood(x1,x2,y,p):
    
    scale, max_succes = p
    
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        X1,X2 = np.meshgrid(x1,x2)
        q = 1/(1+np.exp(-(X1-X2)/scale))
    else:
        q = 1/(1+np.exp(-(x1-x2)/scale))
    
    return binom(max_succes,y) * q**y * (1-q)**(max_succes-y)

