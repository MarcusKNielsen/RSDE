import matplotlib.pyplot as plt
from src.hermite import nodes,vander
import numpy as np
from scipy.stats import norm,gumbel_r,logistic

def dealiased_hermite_product(z,u1,u2,V,Vinv,Prob=False):
    
    N = z.shape[0]
    M = 3*N/2
    M = np.ceil(M)
    M = int(M)
    
    u1hat_pad = np.zeros(M,dtype=np.complex128)
    u2hat_pad = np.zeros(M,dtype=np.complex128)
    
    u1hat_pad[:N] = Vinv @ u1
    u2hat_pad[:N] = Vinv @ u2
    
    z_pad,_ = nodes(M,Prob=Prob)
    V_pad,_ = vander(z_pad)
    V_pad_inv = np.linalg.inv(V_pad)
    
    u1_pad = V_pad@u1hat_pad
    u2_pad = V_pad@u2hat_pad    
    
    u1u2_pad = u1_pad*u2_pad
    
    u1u2hat_pad = V_pad_inv@u1u2_pad
    
    u1u2hat = u1u2hat_pad[:N]
    
    product = V@u1u2hat
    
    return product

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

    # ad hoc adjustments to solution: remove signs and normalize
    w = np.abs(w)
    w = w/mass
        
    if np.abs(mass - 1) > 0.01:
        print(f"Integral after convolution is far from normalized: {mass}")

    if np.abs(mean) > 0.01:
        print(f"mean after convolution is not zero: {mean}")
        
    if np.abs(var - 1) > 0.01:
        print(f"variance after convolution is not one: {var}")

    # Init array for results
    y = np.zeros(N+2)
    
    # Save all results
    y[0]  = m
    y[1]  = s
    y[2:] = np.sqrt(w)
    
    return y


N = 32
z,w = nodes(N,Prob=True)
V,Vz = vander(z,HermiteFunc=True,Prob=True)
Vinv = np.linalg.inv(V)
Mz = Vinv.T @ Vinv

# For standard gauss
bhat = np.zeros(N)
bhat[0] = 1

# player 1
m1,s1 = 2.0 , 3.0
y1 = np.zeros(N+2)
y1[0] = m1
y1[1] = s1
y1[2:] = V@bhat

# player 2
m2,s2 = -2.0 , 1.5
y2 = np.zeros(N+2)
y2[0] = m2
y2[1] = s2
y2[2:] = V@bhat

y = hermite_convolve(z,y1,y2,V,Vinv)
m = y[0]
s = y[1]
b = y[2:]
w = b*b
u = w/s
x = s*z+m

# Compute convolution using standard method
xlarge = np.linspace(np.min(x)-20, np.max(x)+20, 3000)
dx = xlarge[1] - xlarge[0]
p1 = norm.pdf(xlarge,m1,s1)
p2 = norm.pdf(xlarge,m2,s2)
res = dx * np.convolve(p1, p2, mode='same')

zlarge = np.linspace(np.min(z), np.max(z), 300)
Vlarge,_ = vander(zlarge,N,Prob=True)
blarge = Vlarge@Vinv@b
ularge = blarge**2/s

plt.figure()
plt.plot(xlarge,res,label="Ground Truth")
#plt.plot(x,u,".")
plt.plot(s*zlarge+m,ularge,"--")
plt.xlim([-15,15])
plt.grid(True)
plt.legend()
plt.show()

print(np.trapz(res, xlarge))  # Convolution method
print(np.sum(Mz@w))  # Spectral method

#%%

N = 16
z,w = nodes(N,Prob=True)
V,Vz = vander(z,HermiteFunc=True,Prob=True)
Vinv = np.linalg.inv(V)
Mz = Vinv.T @ Vinv

zlarge = np.linspace(-50,50,3000)
p1,p2 = 0,2
gumbel = gumbel_r.pdf(zlarge,p1,p2)
dz = zlarge[1] - zlarge[0]
m1 = np.sum(zlarge*gumbel*dz)
s1 = np.sqrt(np.sum((zlarge-m1)**2 * gumbel * dz))
x1 = s1*z+m1
u1 = gumbel_r.pdf(x1,p1,p2)
w1 = s1*u1
y1 = np.zeros(N+2)
b1 = np.sqrt(w1)
y1[0] = m1
y1[1] = s1
y1[2:] = b1

# player 2
bhat = np.zeros(N)
bhat[0] = 1
b2 = V@bhat
m2,s2 = -5.0 , 9.5
y2 = np.zeros(N+2)
y2[0] = m2
y2[1] = s2
y2[2:] = b2
w2 = b2*b2
u2 = w2/s2
x2 = s2*z+m2

gauss = norm.pdf(zlarge,m2,s2)
res2 = dz * np.convolve(gauss, gumbel, mode='same')

y = hermite_convolve(z,y1,y2,V,Vinv)
m = y[0]
s = y[1]
b = y[2:]
w = b*b
u = w/s
x = s*z+m

zinter = np.linspace(-10,10,400)
V1,_ = vander(zinter,N,Prob=True)
binter = V1 @ Vinv @ b
uinter = binter**2 / s
xinter = s*zinter + m

print(np.trapz(res2, zlarge))
print(np.sum(Mz@w))

fig,ax = plt.subplots(1,2,figsize=(10,4))

ax[0].plot(zlarge,gumbel,label="Gumbel")
ax[0].plot(zlarge,gauss,label="Gauss")
ax[0].plot(x1,u1,".")
ax[0].plot(x2,u2,".")

ax[1].plot(zlarge,res2,label="Convolution")
ax[1].plot(x,u,".")
ax[1].plot(xinter,uinter,"--")
#ax[1].set_xlim([m-15,m+15])

plt.legend()
plt.tight_layout()
plt.show()

#%%

N = 16
z,w = nodes(N,Prob=True)
V,Vz = vander(z,HermiteFunc=True,Prob=True)
Vinv = np.linalg.inv(V)
Mz = Vinv.T @ Vinv

zlarge = np.linspace(-15,15,3000)
dz = zlarge[1] - zlarge[0]
p1,p2,s = 0,-2,1
gumbel1 = gumbel_r.pdf(zlarge,p1,s)
gumbel2 = gumbel_r.pdf(zlarge,p2,s)

res3 = dz * np.convolve(gumbel1, gumbel2[::-1], mode='same')
print(np.trapz(res3, zlarge))

test = logistic.pdf(zlarge,p2-p1,s)

# setup players
m1 = np.sum(zlarge*gumbel1*dz)
s1 = np.sqrt(np.sum((zlarge-m1)**2 * gumbel1 * dz))
x1 = s1*z+m1
u1 = gumbel_r.pdf(x1,p1,s)
w1 = s1*u1
y1 = np.zeros(N+2)
b1 = np.sqrt(w1)
y1[0] = m1
y1[1] = s1
y1[2:] = b1

m2 = np.sum(zlarge*gumbel2*dz)
s2 = np.sqrt(np.sum((zlarge-m2)**2 * gumbel2 * dz))
x2 = s2*z+m2
u2 = gumbel_r.pdf(x2,p2,s)
w2 = s2*u2
y2 = np.zeros(N+2)
b2 = np.sqrt(w2)
y2[0] = m2
y2[1] = s2
y2[2:] = b2

y = hermite_convolve(z,y2,y1,V,Vinv,diff=True)
m = y[0]
s = y[1]
b = y[2:]
w = b*b
u = w/s
x = s*z+m

zinter = np.linspace(-10,10,400)
V1,_ = vander(zinter,N,Prob=True)
binter = V1 @ Vinv @ b
uinter = binter**2 / s
xinter = s*zinter + m

fig,ax = plt.subplots(1,2,figsize=(10,4))

ax[0].plot(zlarge,gumbel1,label="Gumbel1")
ax[0].plot(zlarge,gumbel2,label="Gumbel2")
ax[0].legend()

#ax[1].plot(zlarge,res3,label="Convolution")
ax[1].plot(zlarge,test,linewidth=3.0,label="logistic (theory)")
ax[1].plot(xinter,uinter,"--",color="red",label="interpolant")
ax[1].plot(x,u,".",label="nodal solution")

plt.legend()
plt.tight_layout()
plt.show()




