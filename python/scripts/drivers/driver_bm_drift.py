import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander
from scipy.integrate import solve_ivp
from src.ivp_solver import ivp_solver, fun, Jac
from scripts.systems.bm_drift import a,D,dadx,dDdx

"""
Brownian Motion with Drift
dXt = p1*dt+p2*dBt
"""

def gauss(t,x,a,D,loc):
    return np.exp(-(x-a*t-loc)**2/(4*D*t))/np.sqrt(4*np.pi*D*t)

N = 32
z,w = nodes(N)
V,Vz = vander(z)

Vinv = np.linalg.inv(V)

Dz = Vz @ Vinv
Dz2 = Dz@Dz
Mz = (Vinv.T @ Vinv).T

t0 = 10.0
tf = 20.0
p = np.array([0.0,1.0])

# initial condition (skew-gauss)
x0 = np.linspace(-2*t0*p[1],2*t0*p[1],1000)
loc = -p[0]*t0
u0 = gauss(t0,x0,p[0],p[1]**2/2,loc)
dx = x0[1] - x0[0]
y0 = np.zeros(N+2)
y0[0]  = np.sum(x0*u0*dx) 
y0[1]  = np.sqrt(np.sum((x0-y0[0])**2*u0*dx)) 
y0[2:] = y0[1]*gauss(t0,y0[1]*z+y0[0],p[0],p[1]**2/2,loc)


tspan=[t0, tf]
p1 = (z, Dz, Dz2, Mz, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
res = ivp_solver(fun, Jac, tspan, y0, pfun=p1, pjac=p2)

mf = res['y'][0]
sf = res['y'][1]
wf = res['y'][2:]
xf = sf*z+mf
uf = wf/sf

x = np.linspace(np.min(xf),np.max(xf),100)
plt.figure()
plt.plot(x0,u0,label=r"$u(x,t_0)$")
plt.plot(x,gauss(tf,x,p[0],p[1]**2/2,loc),label="analytical")
plt.plot(xf,uf,".",label=r"$u(x,t_f)$")
plt.xlabel("x: space")
plt.title("Initial condition and stationary distribution")
plt.legend()
plt.tight_layout()
plt.show()

error = gauss(tf,xf,p[0],p[1]**2/2,loc) - uf
print(f"error = {np.max(np.abs(error))}")
plt.figure()
plt.plot(xf,error)
plt.show()

#%%

J = Jac(res['t'], res['y'], *p2)
#J = J[3:,3:]
eigs = np.linalg.eigvals(J)
Re = np.real(eigs)
Im = np.imag(eigs)
plt.figure()
plt.plot(Re,Im,".")
plt.grid(True)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.show()
Re = np.sort(Re)
print(Re.max())

#%%

from scipy.optimize import approx_fprime

def compute_jacobian(fun, t, y, z, Dz, Dz2, M, a, D, p):
    """
    Computes the Jacobian of the function `fun` with respect to y using finite differences.
    """
    
    epsilon=1e-10
    
    
    def wrapped_fun(y_flat):
        y_vec = y_flat.reshape(y.shape)
        return fun(t, y_vec, z, Dz, Dz2, M, a, D, p).flatten()
    
    jacobian = approx_fprime(y.flatten(), wrapped_fun, epsilon)
    return jacobian.reshape(y.size, y.size)

J_test = compute_jacobian(fun, res['t'], res['y'], *p1)



#plt.close('all')








