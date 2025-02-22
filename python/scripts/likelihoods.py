import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander,hermite_weight_matrix


"""
Construct priors for two players
"""

mu1,s1 =  1.5, 0.5
mu2,s2 = -0.5, 0.6

N = 24
z,w = nodes(N)
V,Vz = vander(z)

Vinv = np.linalg.inv(V)
M = (Vinv.T @ Vinv).T

what = np.zeros(N)
what[0] = 1/(np.sqrt(2)*np.pi**(0.25))

w1 = V@what
w2 = V@what

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

zlarge = np.linspace(-7.5,7.5,300)
Vlarge,_ = vander(zlarge,N)

x1 = s1*zlarge+mu1
x2 = s2*zlarge+mu2

u1 = (V@what)/s1
u2 = (V@what)/s2

w1large = Vlarge@what
w2large = Vlarge@what

u1large = w1large/s1
u2large = w2large/s2

axes[0].plot(x1,u1large,label=r"$u_1(t,x_1)$")
axes[0].set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"Density: $u_1$")
axes[0].set_title("Prior Player 1")
axes[0].legend()

axes[1].plot(x2,u2large,label=r"$u_2(t,x_2)$")
axes[1].set_xlabel(r"$x_2$")
axes[1].set_ylabel(r"Density: $u_2$")
axes[1].set_title("Prior Player 2")
axes[1].legend()

U = np.outer(u1large, u2large)
X1,X2 = np.meshgrid(x1,x2)
axes[2].pcolormesh(X1,X2,U)
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

x1_small = s1*z1+mu1
x2_small = s2*z2+mu2

X1,X2 = np.meshgrid(x1_small,x2_small)


"""
Setup State Likelihood
"""


# Create meshgrid
yl,yr = -10,10
y = np.linspace(yl, yr, 1000)
Y1, Y2 = np.meshgrid(y, y)

# Compute Ay and P
def like(Y1,Y2):
    A = np.array([[1, -1], [0.0, -0.0], [-1, 1]])
    Y_stack = np.stack([Y1, Y2], axis=-1)
    Ay = np.einsum('kj, nmj -> knm', A, Y_stack)
    P = np.exp(Ay)
    Z = np.sum(P, axis=0)
    P = P / Z
    return P

P = like(Y1,Y2)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

titles = ["Player 1 wins", "Draw", "Player 2 wins"]

# Define common tick positions
tick_positions = np.linspace(yl, yr, 5)  # Adjust number of ticks if needed

# Iterate over the first 3 indices (i=0,1,2)
for i in range(3):
    pcm = axes[i].pcolormesh(Y1, Y2, P[i], shading='auto')
    axes[i].set_title(titles[i])
    axes[i].set_aspect('equal')
    axes[i].plot(X1,X2,".",color="red",markersize=1)

    # Ensure all subplots have the same axis limits and tick positions
    axes[i].set_xlim([-3, 3])
    axes[i].set_ylim([-3, 3])
    axes[i].set_xticks(tick_positions)
    axes[i].set_yticks(tick_positions)
    axes[i].set_xlabel(r"$x_1$")
    axes[i].set_ylabel(r"$x_2$")

    # Force y-axis ticks to be visible on all subplots
    axes[i].tick_params(left=True, labelleft=True)

    # Add colorbar
    fig.colorbar(pcm, ax=axes[i], fraction=0.046, pad=0.04)

plt.suptitle("State Likelihood Function")
plt.tight_layout()
plt.show()


#%%
P_test = like(X1,X2)
v = P_test[1]

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
Compute integral
"""

Vv,_ = vander(z1,HermiteFunc=False) 
Vvlarge,_ = vander(zlarge,N,HermiteFunc=False) 

Vvinv = np.linalg.inv(Vv)

W = hermite_weight_matrix(N1,N)

i = -1

# fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# axes[0].plot(x1_small,v[i])
# axes[0].plot(x1_small,u1)
# axes[0].plot(x1_small,u1*v[i])
# axes[0].set_xlabel(r"$x_1$")

# axes[1].plot(z1,v[i])
# axes[1].plot(z1,w1/s1)
# axes[1].plot(z1,(w1/s1)*v[i])
# axes[1].set_xlabel(r"$z_1$")

# plt.tight_layout()
# plt.show()

dz = zlarge[1:] - zlarge[:-1]

v1large = Vvlarge@(Vvinv @ v[i])
int1 = np.sum(v1large[1:]*w1large[1:]*dz)

v2large = Vvlarge@(Vvinv @ (v.T)[i])
int2 = np.sum(v2large[1:]*w2large[1:]*dz)

# print(int1,int2)


# fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# axes[0].plot(zlarge,w1large)
# axes[0].plot(zlarge,v1large)
# axes[0].set_xlabel(r"$z_1$")

# axes[1].plot(zlarge,w2large)
# axes[1].plot(zlarge,v2large)
# axes[1].set_xlabel(r"$z_2$")

# plt.tight_layout()
# plt.show()

#%%

like1 = (Vvinv @ (v.T)).T @ W @ what
like2 = (Vvinv @ v).T @ W @ what

# like1_large = Vvlarge @ (Vvinv@like1)
# like2_large = Vvlarge @ (Vvinv@like2)

post1 = like1*w1
post1 = post1/np.sum(M@post1,axis=0)
post1 = post1/s1
post1_large = Vlarge @ (Vinv@post1)

post2 = like2*w2
post2 = post2/np.sum(M@post2,axis=0)
post2 = post2/s2
post2_large = Vlarge @ (Vinv@post2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x1,u1large,label=r"prior")
#axes[0].plot(x1,like1_large,label=r"likelihood")
axes[0].plot(x1_small,like1,label=r"likelihood")
axes[0].plot(x1,post1_large,label=r"posterior")
axes[0].set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"Density: $u_1$")
axes[0].set_title("Player 1")
axes[0].legend()

axes[1].plot(x2,u2large,label=r"prior")
#axes[1].plot(x2,like2_large,label=r"likelihood")
axes[1].plot(x2_small,like2,label=r"likelihood")
axes[1].plot(x2,post2_large,label=r"posterior")
axes[1].set_xlabel(r"$x_2$")
axes[1].set_ylabel(r"Density: $u_2$")
axes[1].set_title("Player 2")
axes[1].legend()

U = np.outer(u1large, u2large)
X1,X2 = np.meshgrid(x1,x2)
axes[2].pcolormesh(X1,X2,U)
axes[2].set_title(r"Prior: $u_1(t,x_1)u_2(t,x_2)$")
axes[2].set_xlabel(r"$x_1$")
axes[2].set_ylabel(r"$x_2$")

plt.tight_layout()
plt.show()






