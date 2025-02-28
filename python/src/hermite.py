import numpy as np
from numpy.polynomial.hermite import hermgauss

def nodes(N, Prob=False):
    nodes, weights = hermgauss(N)
    
    if Prob == False:
        return nodes, weights
    
    # Convert to probabilistic Hermite-Gauss quadrature
    nodes = nodes / np.sqrt(2)
    weights = weights / np.sqrt(np.pi)
    
    return nodes, weights
    

def vander( x, N=None, HermiteFunc=True, Prob=False):

    K = len(x)
    
    if N == None:
        N = K
    
    # Initialize vandermonde matrices
    V  = np.zeros([K,N])
    Vx = np.zeros([K,N])    

    if Prob==False:
        
        # Hermite (n=0)
        V[:,0]  = np.pi**(-0.25)
        
        if HermiteFunc == True:
            V[:,0]  *= np.exp(-0.5*x**2)
    
        # Diff Hermite (n=0)
        Vx[:,0] = -x*V[:,0]
    
        if N == 1:
            return V,Vx
    
        # Hermite (n=1)
        V[:,1]  = np.sqrt(2)*x*V[:,0]
        
        # Diff Hermite (n=1)
        Vx[:,1] = 2*np.sqrt(1/2)*V[:,0]-x*V[:,1]
        
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
        
    else:
            
        # Hermite (n=0)
        V[:,0]  = (2*np.pi)**(-0.25)
        
        if HermiteFunc == True:
            V[:,0]  *= np.exp(-0.25*x**2)
    
        # Diff Hermite (n=0)
        Vx[:,0] = -x*V[:,0]
    
        if N == 1:
            return V,Vx
    
        # Hermite (n=1)
        V[:,1]  = x*V[:,0]
        
        # Diff Hermite (n=1)
        Vx[:,1] = 2*np.sqrt(1/2)*V[:,0]-x*V[:,1]
        
        if N == 2:
            return V,Vx
        
        for n in range(1,N-1):
            
            # Recurrence relation Hermite
            V[:,n+1]  = np.sqrt(2/(n+1)) * (x*V[:,n] - np.sqrt(n/2)*V[:,n-1])
            
            # Recurrence relation Diff Hermite
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

    """
    Discrete Product Rule
    """
    
    import matplotlib.pyplot as plt
    
    N = 50
    z,w = nodes(N)
    V,Vz = vander(z)
    Vinv = np.linalg.inv(V)
    Dz =  Vz @ Vinv
    
    f = np.exp(-z**2/5)
    g = np.exp(-z**2/3)
    dfg = (-16*z/15) * f * g
    
    res1 = Dz@(f*g)
    res2 = (Dz@f)*g + f*(Dz@g)
    
    fig,axes = plt.subplots(1,2,figsize=(10,5))
    
    axes[0].plot(z,f,".-",label=r"$f(z) = \exp(-z^2/5)$")
    axes[0].plot(z,g,".-",label=r"$g(z) = \exp(-z^2/3)$")
    axes[0].plot(z,f*g,".-",label=r"$fg$")
    axes[0].legend()
    
    axes[1].plot(z,res1,".-",label=r"$\mathcal{D}_z(f \cdot g)$")
    axes[1].plot(z,res2,".-",label=r"$(\mathcal{D}_zf) \cdot g + f \cdot (\mathcal{D}_zg)$")
    axes[1].plot(z,dfg,"--",label=r"$ \frac{\partial}{\partial z} fg $")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    