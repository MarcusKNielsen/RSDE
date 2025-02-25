import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def likelihood(x):
    return norm.pdf(x,2.0)



m1 = 0
s1 = 1.144

x = np.linspace(m1-7*s1,m1+7*s1,10000)
dx = x[1] - x[0]

u = norm.pdf(x,m1,s1)
like = likelihood(x)

unew = like*u
print(np.sum(unew*dx))
unew = unew/np.sum(unew*dx)

mnew = np.sum(x*unew*dx)
snew = np.sqrt(np.sum((x-mnew)**2*unew*dx))

#%%
from src.hermite import nodes,vander

# Initialize grid and matrices
N = 32
z,w = nodes(N)

# Matrices based on Hermite Functions
V,Vz = vander(z)
Vinv = np.linalg.inv(V)
Mz = (Vinv.T @ Vinv).T

# initial condition
y = np.zeros(N+2)
y[0] = m1
y[1] = s1
what = np.zeros(N)
what[0] = 1/(np.sqrt(2*np.pi) * (np.pi)**(-0.25))
y[2:] = V@what



w1 = y[2:]
x1 = s1*z+m1
like1 = likelihood(x1)

# update
u1 = like1*(w1/s1)
int1 = s1*np.sum(Mz@u1,axis=0)
print(int1)
u1 = u1/int1


#%%

fig,ax = plt.subplots(1,3,figsize=(10,4))

ax[0].plot(x,u,label="u")
ax[0].plot(x1,w1/s1,".",label="w/s")
ax[0].set_title("prior")
ax[0].legend()

ax[1].plot(x,like,label="like")
ax[1].plot(x1,like1,".",label="like1")
ax[1].set_title("likelihood")
ax[1].legend()

ax[2].plot(x,unew,label="unew")
ax[2].plot(x1,u1,".",label="u1")
test = norm.pdf(x,mnew,snew)
ax[2].plot(x,test,"--",label="test")
ax[2].set_title("posterior")
ax[2].legend()

plt.tight_layout()
plt.show()



