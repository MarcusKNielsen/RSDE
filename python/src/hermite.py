import numpy as np
from numpy.polynomial.hermite import hermgauss

def nodes(N):
    nodes, weights = hermgauss(N)
    weights *= np.sqrt(np.pi)
    return nodes, weights

def vander(x,N=None):

    K = len(x)
    
    if N == None:
        N = K
    
    # Initialize vandermonde matrices
    V  = np.zeros([K,N])
    Vx = np.zeros([K,N])
    
    # Hermite Function (n=0)
    V[:,0]  = np.pi**(-0.25) * np.exp(-0.5*x**2)

    # Diff Hermite Function (n=0)
    Vx[:,0] = -x*V[:,0]

    if N == 1:
        return V,Vx

    # Hermite Function (n=1)
    V[:,1]  = np.sqrt(2)*x*V[:,0]
    
    # Diff Hermite Function (n=1)
    Vx[:,1] = 2*np.sqrt(1/2)*V[:,0]-x*V[:,1]
    
    if N == 2:
        return V,Vx
    
    for n in range(1,N-1):
        
        # Recurrence relation Hermite Function
        V[:,n+1]  = np.sqrt(2/(n+1)) * (x*V[:,n] - np.sqrt(n/2)*V[:,n-1])
        
        # Recurrence relation Diff Hermite Function
        Vx[:,n+1] = 2*np.sqrt((n+1)/2)*V[:,n] - x*V[:,n+1]
        
    return V,Vx


    




