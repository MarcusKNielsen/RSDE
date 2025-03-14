import matplotlib.pyplot as plt
from src.hermite import nodes,vander
import numpy as np
from scipy.stats import norm

# Initialize grid and matrices
N = 16
z,w = nodes(N,Prob=True)

# Matrices based on Hermite Functions
V,Vz = vander(z,Prob=True)
Vinv = np.linalg.inv(V)
Mz = Vinv.T @ Vinv

y = np.zeros(N+2)
y[1] = np.sqrt(2)
bhat = np.zeros(N)
bhat[0] = 1
y[2:] = V@bhat

# Two standard gauss
b1 = V@bhat
b2 = V@bhat

x = np.linspace(np.min(z),np.max(z),1000)
std_gauss = norm.pdf(x,0,1)
convolve_gauss = norm.pdf(x,0,np.sqrt(2))

#%%

M = N
zinter,_ = nodes(M)
Vinter,_ = vander(zinter,N,Prob=True)

b1inter = Vinter @ Vinv @ b1
b2inter = Vinter @ Vinv @ b2

w1inter = b1inter**2
w2inter = b2inter**2

V1,_ = vander(zinter)
V1inv = np.linalg.inv(V1)
M1z = V1inv.T @ V1inv

n = np.arange(M)
F = V1 @ np.diag((-1j)**n) @ V1inv

Finv = V1 @ np.diag((1j)**n) @ V1inv

Fw1 = F@w1inter
Fw2 = F@w2inter

Fw = np.sqrt(2*np.pi)*(Fw1*Fw2)

w1_convolve_w2 = (Finv@Fw).real

fig,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].plot(x,std_gauss)
ax[0].plot(zinter,Fw1.real,".")
ax[0].plot(zinter,Fw2.real,".")
ax[0].grid()

ax[1].plot(x/np.sqrt(2),np.sqrt(2)*convolve_gauss)
ax[1].plot(zinter,w1_convolve_w2,".")
ax[1].plot(x,std_gauss,"--")
ax[1].grid()

plt.tight_layout()
plt.show()


print(np.sum(M1z@w1_convolve_w2))  # Spectral method


#%%





zlarge = np.linspace(-15,15,1000)
Vlarge,_ = vander(zlarge,N,Prob=True)

b1_large = Vlarge @ (Vinv @ b1)
b2_large = Vlarge @ (Vinv @ b2)

w1_large = b1_large**2
w2_large = b2_large**2

dz = zlarge[1] - zlarge[0]
result = dz * np.convolve(w1_large, w2_large, mode='same')

plt.figure()
plt.plot(zlarge,result,label="convolve")
plt.plot(zinter,w1_convolve_w2,".",label="spectral")
plt.legend()
plt.show()

print(np.trapz(result, zlarge))  # Convolution method
print(np.sum(M1z@w1_convolve_w2))  # Spectral method

#%%

"""
This code works
"""

M = 64
zinter,_ = nodes(M)
Vinter,_ = vander(zinter,N,Prob=True)

b1inter = Vinter @ Vinv @ b1
b2inter = Vinter @ Vinv @ b2

w1inter = b1inter**2
w2inter = b2inter**2

V1,_ = vander(zinter)
V1inv = np.linalg.inv(V1)
M1z = V1inv.T @ V1inv

n = np.arange(M)
F = V1 @ np.diag((-1j)**n) @ V1inv
Finv = V1 @ np.diag((1j)**n) @ V1inv

Fw1 = F@w1inter
Fw2 = F@w2inter

product = Fw1*Fw2

w1_convolve_w2 = np.sqrt(2*np.pi)*(Finv@product).real

zlarge = np.linspace(-15,15,1000)
Vlarge,_ = vander(zlarge,N,Prob=True)

b1_large = Vlarge @ (Vinv @ b1)
b2_large = Vlarge @ (Vinv @ b2)

w1_large = b1_large**2
w2_large = b2_large**2

dz = zlarge[1] - zlarge[0]
result = dz * np.convolve(w1_large, w2_large, mode='same')

plt.figure()
plt.plot(zlarge,result,label="convolve")
plt.plot(zinter,w1_convolve_w2,".",label="spectral")
plt.legend()
plt.show()

print(np.trapz(result, zlarge))  # Convolution method
print(np.sum(M1z@w1_convolve_w2))  # Spectral method


