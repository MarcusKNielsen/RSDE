import numpy as np
from numpy.polynomial.hermite import hermgauss

def nodes(N):
    nodes, weights = hermgauss(N)
    return nodes, weights

def vander(x,N=None,HermiteFunc=True):

    K = len(x)
    
    if N == None:
        N = K
    
    # Initialize vandermonde matrices
    V  = np.zeros([K,N])
    Vx = np.zeros([K,N])
    
    # Hermite Function (n=0)
    V[:,0]  = np.pi**(-0.25)
    
    if HermiteFunc == True:
        V[:,0]  *= np.exp(-0.5*x**2)

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

def hermite_weight_matrix(N,M=None):
    
    if M == None:
        M=N

    y,w = nodes(N if N > M else M)
    VN,_ = vander(np.sqrt(2)*y,N,HermiteFunc=False)
    VM,_ = vander(np.sqrt(2)*y,M,HermiteFunc=False)

    W = np.zeros([N,M])

    for i in range(N):
        for j in range(M):
            W[i,j] = np.sqrt(2)*np.sum(VN[:,i]*VM[:,j]*w)

    return W



if __name__ == "__main__":
    
    
    """
    The following code can be used to test hermite_weight_matrix correctness.
    For large values of N and M, then the two methods seems to disagree,
    probably because of large and small values in the expression v1*v2*np.exp(-x**2/2).
    """
    N = 7
    M = 28

    x = np.linspace(-26,26,10000)
    dx = x[1] - x[0]
    VNlarge,_ = vander(x,N,HermiteFunc=False)
    VMlarge,_ = vander(x,M,HermiteFunc=False)
    W = np.zeros([N,M])

    for i in range(N):
        for j in range(M):
            v1 = np.zeros(N)
            v2 = np.zeros(M)
            v1[i] = 1
            v2[j] = 1
            v1 = VNlarge@v1
            v2 = VMlarge@v2
            
            W[i,j] = np.sum(v1*v2*np.exp(-x**2/2)*dx)



    W1 = hermite_weight_matrix(N,M) 

    print(np.max(np.abs(W-W1)))

