import numpy as np

"""
Golub-Welsch algorithm
Function to compute Gauss-Hermite quadrature nodes and weights.
Both the physicist version and the probabilistic version.
"""

def nodes(N,Prob=False):
    
    # subdiagonals
    if Prob == True:
        sub_diags = np.sqrt(np.arange(1, N))
    else:
        sub_diags = np.sqrt(np.arange(1, N)/2)
    
    # Construct the symmetric tridiagonal matrix
    J = np.diag(sub_diags, 1) + np.diag(sub_diags, -1)
    
    # Compute eigenvalues and eigenvectors
    nodes, V = np.linalg.eigh(J)
    
    # Sort eigenvalues and corresponding eigenvectors
    i = np.argsort(nodes)
    nodes = nodes[i]
    Vtop = V[0, :]
    Vtop = Vtop[i]
    
    # Compute weights
    if Prob == True:
        weights = np.sqrt(np.pi/2) * Vtop ** 2
    else:
        weights = np.sqrt(np.pi) * Vtop ** 2
    
    return nodes, weights
    

def vander(x,N=None,HermiteFunc=True,Prob=False):
    
    K = len(x)
    
    if N == None:
        N = K
    
    # Initialize vandermonde matrices
    V  = np.zeros([K,N])
    Vx = np.zeros([K,N])    
    
    if Prob == True:
        # Hermite (n=0)
        V[:,0]  = (2*np.pi)**(-0.25)
        Vx[:,0] = np.zeros_like(x)
    
        if N == 1:
            return V,Vx
    
        # Hermite (n=1)
        V[:,1]  = x*V[:,0]
        Vx[:,1] = V[:,0]
        
        if N == 2:
            return V,Vx
        
        for n in range(1,N-1):
            
            # Recurrence relation for probabilistic Hermite polynomials
            V[:,n+1]  = np.sqrt(1/(n+1)) * (x*V[:,n] - np.sqrt(n)*V[:,n-1])
            
            # Recurrence relation derivative of probabilistic Hermite polynomials
            Vx[:,n+1] = np.sqrt(n+1)*V[:,n]
    
        # Convert Hermite polynomials to Hermite functions (probabilistic version)
        if HermiteFunc==True:
            X = np.diag(x)
            gauss = np.diag(np.exp(-0.25*x**2))
            V  = gauss @ V
            Vx = gauss@Vx-0.5*X@V
    
    else:
        # Hermite (n=0)
        V[:,0]  = np.pi**(-0.25)
        Vx[:,0] = np.zeros_like(x)

        if N == 1:
            return V,Vx

        # Hermite (n=1)
        V[:,1]  = np.sqrt(2)*x*V[:,0]
        Vx[:,1] = np.sqrt(2)*np.pi**(-0.25)
        
        if N == 2:
            return V,Vx
        
        for n in range(1,N-1):
            
            # Recurrence relation for physicist Hermite polynomials
            V[:,n+1]  = np.sqrt(2/(n+1)) * (x*V[:,n] - np.sqrt(n/2)*V[:,n-1])
            
            # Recurrence relation derivative of physicist Hermite polynomials
            Vx[:,n+1] = np.sqrt(2*(n+1)) * V[:,n]
            
        # Convert Hermite polynomials to Hermite functions (physicist version)
        if HermiteFunc==True:
            X = np.diag(x)
            gauss = np.diag(np.exp(-0.5*x**2))
            V  = gauss @ V
            Vx = gauss@Vx-X@V
            
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

def hermite_convolve(z_prob,y1,y2,V_prob,V_prob_inv,M=128,diff=False):
    
    # if diff true  then Y1+Y2
    # if diff False then Y1-Y2
    
    # Extract player information from state vectors
    m1 = y1[0]
    s1 = y1[1]
    b1 = y1[2:]

    m2 = y2[0]
    s2 = y2[1]
    b2 = y2[2:]
    
    # Compute mean and standard deviation of convolution
    m = (m1+m2) if not diff else (m1-m2)
    s = np.sqrt(s1**2 + s2**2)  
  
    # Setup grid needed for Fourier transform
    N = z_prob.shape[0]
    z_phys,_ = nodes(M)
    V_prob_to_phys,_ = vander(z_phys,N,Prob=True)
    
    # Convert from Probabilistic Hermite to Physicist Hermite representation
    C_prob_to_phys = V_prob_to_phys @ V_prob_inv
    
    b1_phys = C_prob_to_phys @ b1
    b2_phys = C_prob_to_phys @ b2
    
    # Compute interpolated standardized densities w1 and w2
    w1 = b1_phys**2
    w2 = b2_phys**2
    
    # Compute vandermonde matrices, based on physicist hermite, for Fourier matrix
    V_phys,_ = vander(z_phys)
    V_phys_inv = np.linalg.inv(V_phys)
    
    # Setup Fourier matrices based eigendecomposition
    n = np.arange(M)
    F = V_phys @ np.diag((-1j)**n) @ V_phys_inv
    Finv = V_phys @ np.diag((1j)**n) @ V_phys_inv
    
    # Compute Fourier transform of w1 and w2
    Fw1 = (F @ w1) if not diff else (Finv @ w1)
    Fw2 = F@w2
    
    V1eval,_ = vander(s1*z_phys,M)
    V2eval,_ = vander(s2*z_phys,M)
    
    Fw1 = V1eval@V_phys_inv@Fw1
    Fw2 = V2eval@V_phys_inv@Fw2
    
    # Compute product
    product = Fw1*Fw2
    
    # Compute convolution using convolution theorem
    Fw = np.sqrt(2*np.pi)*product    
    w = (Finv@Fw).real
    
    # Convert from Physicist Hermite to Probabilistic Hermite and adjust for scale
    Veval,_ = vander(s*z_prob,M)
    w = Veval @ V_phys_inv @ w
    w = s*w

    # Quality control
    Mz = V_prob_inv.T @ V_prob_inv
    mass = np.sum(Mz@w)
    mean = z_prob@Mz@w
    var  = z_prob**2@Mz@w
        
    if np.abs(mass - 1) > 0.01:
        print(f"mass after convolution is far from normalized: {mass}")

    if np.abs(mean) > 0.01:
        print(f"mean after convolution is not zero: {mean}")
        
    if np.abs(var - 1) > 0.01:
        print(f"variance after convolution is not one: {var}")

    # ad hoc adjustments to solution: remove signs and normalize
    w = np.abs(w)
    w = w/mass

    # Init array for results
    y = np.zeros(N+2)
    
    # Save all results
    y[0]  = m
    y[1]  = s
    y[2:] = np.sqrt(w)
    
    return y

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
    z,w = nodes(N,Prob=True)
    V,Vz = vander(z,HermiteFunc=True,Prob=True)
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
    axes[1].set_ylim([-0.65,0.65])
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Example usage
    from numpy.polynomial.hermite import hermgauss
    N = 50
    nodes1, weights1  = nodes(N)
    nodes2, weights2 = hermgauss(N)

    err1 = np.max(np.abs(nodes1 - nodes2))
    err2 = np.max(np.abs(weights1 - weights2))

    print(err1)
    print(err2)
    

