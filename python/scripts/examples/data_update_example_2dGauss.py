import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander,hermite_weight_matrix

def gauss(x,m,s):
    return np.exp(-(x-m)**2/(2*s**2)) / np.sqrt(2*np.pi*s**2)

"""
Construct priors for two players
"""

mu1,s1 =  0.5, 0.5
mu2,s2 = -0.5, 0.6

N = 32
z,w = nodes(N)
V,Vz = vander(z)

Vinv = np.linalg.inv(V)
Mz = (Vinv.T @ Vinv).T

what = np.zeros(N)
what[0] = 1/(np.sqrt(2)*np.pi**(0.25))

w1 = V@what
w2 = V@what

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

zlarge = np.linspace(-7.5,7.5,300)
Vlarge,_ = vander(zlarge,N)

x1 = s1*z+mu1
x2 = s2*z+mu2

x1large = s1*zlarge+mu1
x2large = s2*zlarge+mu2

u1 = (V@what)/s1
u2 = (V@what)/s2

w1large = Vlarge@what
w2large = Vlarge@what

u1large = w1large/s1
u2large = w2large/s2

axes[0].plot(x1large,u1large,label=r"$u_1(t,x_1)$")
axes[0].set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"Density: $u_1$")
axes[0].set_title("Prior Player 1")
axes[0].legend()

axes[1].plot(x2large,u2large,label=r"$u_2(t,x_2)$")
axes[1].set_xlabel(r"$x_2$")
axes[1].set_ylabel(r"Density: $u_2$")
axes[1].set_title("Prior Player 2")
axes[1].legend()

U = np.outer(u1large, u2large)
X1large,X2large = np.meshgrid(x1large,x2large)
axes[2].pcolormesh(X1large,X2large,U)
axes[2].set_title(r"Prior: $u_1(t,x_1)u_2(t,x_2)$")
axes[2].set_xlabel(r"$x_1$")
axes[2].set_ylabel(r"$x_2$")

plt.tight_layout()
plt.show()



#%%

N1 = N
z1,_ = nodes(N1)

N2 = N
z2,_ = nodes(N2)


X1,X2 = np.meshgrid(x1,x2)


"""
Setup State Likelihood
"""


# Create meshgrid
yl,yr = -10,10
y = np.linspace(yl, yr, 1000)
Y1, Y2 = np.meshgrid(y, y)


def like(Y1,Y2):
    return (1 / (2 * np.pi)) * np.exp(-0.5 * (Y1**2 + Y2**2))

P = like(Y1,Y2)

plt.figure()

pcm = plt.pcolormesh(Y1, Y2, P, shading='auto')
plt.plot(X1,X2,".",color="red",markersize=1)

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")


plt.tight_layout()
plt.show()


#%%
v = like(X1,X2)

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
ax.plot_surface(X1, X2, v, cmap='viridis')

# Labels
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('P_test')
plt.tight_layout()
plt.show()


#%%
"""
integrate oppenent out to get a "marginal likelihood"
"""

Vv,_ = vander(z1,HermiteFunc=False) 
Vvinv = np.linalg.inv(Vv)
W = hermite_weight_matrix(N1,N)


like1 = (Vvinv @ (v.T)).T @ W @ what
like2 = (Vvinv @ v).T @ W @ what

#%%
"""
Perform Bayesian update
"""

# Compute Posterior using Bayes Rule
u1 = like1*(w1/s1)
u1 = u1/(s1*np.sum(Mz@u1,axis=0))

u2 = like2*(w2/s2)
u2 = u2/(s2*np.sum(Mz@u2,axis=0))

# Find new mean, standard deviation and w solution
m1 = s1*(x1@Mz@u1)
s1 = np.sqrt(s1*((x1-m1)**2@Mz@u1))
w1 = s1*u1

m2 = s2*(x2@Mz@u2)
s2 = np.sqrt(s2*((x2-m2)**2@Mz@u2))
w2 = s2*u2

u1_large = Vlarge @ (Vinv@u1)
u2_large = Vlarge @ (Vinv@u2)


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x1large,u1large,label=r"prior")
axes[0].plot(x1,like1,label=r"likelihood")
axes[0].plot(x1large,u1_large,label=r"posterior")
axes[0].set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"Density: $u_1$")
axes[0].set_title("Player 1")
axes[0].legend()

axes[1].plot(x2large,u2large,label=r"prior")
axes[1].plot(x2,like2,label=r"likelihood")
axes[1].plot(x2large,u2_large,label=r"posterior")
axes[1].set_xlabel(r"$x_2$")
axes[1].set_ylabel(r"Density: $u_2$")
axes[1].set_title("Player 2")
axes[1].legend()

U = np.outer(u1large, u2large)
axes[2].pcolormesh(X1large,X2large,U)
axes[2].set_title(r"Prior: $u_1(t,x_1)u_2(t,x_2)$")
axes[2].set_xlabel(r"$x_1$")
axes[2].set_ylabel(r"$x_2$")

plt.tight_layout()
plt.show()






