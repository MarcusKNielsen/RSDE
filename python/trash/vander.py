import numpy as np

"""
Old version of vandermonde matrix
"""


def vander(x,N=None,HermiteFunc=True):

    K = len(x)
    
    if N == None:
        N = K
    
    # Initialize vandermonde matrices
    V  = np.zeros([K,N])
    Vx = np.zeros([K,N])    

    # Hermite (n=0)
    V[:,0]  = np.pi**(-0.25)
    
    if HermiteFunc == True:
        V[:,0]  *= np.exp(-0.5*x**2)

    # Diff Hermite (n=0)
    if HermiteFunc == True:
        Vx[:,0] = -x*V[:,0]
    else:
        Vx[:,0] = np.zeros_like(x)

    if N == 1:
        return V,Vx

    # Hermite (n=1)
    V[:,1]  = np.sqrt(2)*x*V[:,0]
    
    # Diff Hermite (n=1)
    if HermiteFunc == True:
        Vx[:,1] = 2*np.sqrt(1/2)*V[:,0]-x*V[:,1]
    else:
        Vx[:,1] = np.sqrt(2)*np.pi**(-0.25)
    
    if N == 2:
        return V,Vx
    
    for n in range(1,N-1):
        
        # Recurrence relation for both Hermite Functions and Polynomials
        V[:,n+1]  = np.sqrt(2/(n+1)) * (x*V[:,n] - np.sqrt(n/2)*V[:,n-1])
        
        if HermiteFunc == True:
            # Recurrence relation Diff Hermite Functions
            Vx[:,n+1] = 2*np.sqrt((n+1)/2)*V[:,n] - x*V[:,n+1]
        else:
            # Recurrence relation Diff Hermite Polynomials
            Vx[:,n+1] = np.sqrt(2*(n+1)) * V[:,n]
        
        
    return V,Vx