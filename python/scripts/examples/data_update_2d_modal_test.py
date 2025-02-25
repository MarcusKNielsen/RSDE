from src.hermite import nodes,vander,hermite_weight_matrix
import matplotlib.pyplot as plt
import numpy as np

def func(Y1,Y2):
    return (1 / (2 * np.pi)) * np.exp(-0.5 * (Y1**2 + Y2**2))
    #return np.sin(Y1)*np.cos(2*Y2)

N = 32
z,w = nodes(N)
V,_ = vander(z) 
Vinv = np.linalg.inv(V)

Z1, Z2 = np.meshgrid(z, z)

What = np.zeros([N,N])
What[0,0] = (1/(np.sqrt(2*np.pi) * (np.pi)**(-0.25)))**2

W = V@What@V.T

W1 = func(Z1,Z2)

fig,ax = plt.subplots(1,2)
ax[0].pcolormesh(Z1, Z2, W, shading='auto')
ax[1].pcolormesh(Z1, Z2, W1, shading='auto')
plt.tight_layout()
plt.show()

W1hat = Vinv @ W1 @ Vinv.T








