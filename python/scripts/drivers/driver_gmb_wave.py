import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander
from src.ivp_solver import ivp_solver, fun_wave, Jac_wave
from scripts.systems.gmb import a,D,dadx,dDdx

"""
Geometric Brownian Motion
dXt = p1*Xt*dt+p2*Xt*dBt
"""

def gbm_pdf(x,t,x0,p1,p2):
    return np.exp(-(np.log(x) - np.log(x0) - (p1-0.5*p2**2)*t)**2/(2*p2**2*t))/(x*p2*np.sqrt(2*np.pi*t))


N = 16
z,w = nodes(N,Prob=True)
V,Vz = vander(z,Prob=True)
Vinv = np.linalg.inv(V)
Dz = Vz @ Vinv
Dz2 = Dz@Dz
Mz = (Vinv.T @ Vinv).T

t0 = 0.1
tf = t0 + 1e-9
p = np.array([2.1,0.1])

loc = 5
x0 = np.linspace(4,8,1000)
u0 = gbm_pdf(x0,t0,loc,p[0],p[1])
dx = x0[1] - x0[0]
y0 = np.zeros(N+2)
y0[0]  = np.sum(x0*u0*dx) 
y0[1]  = np.sqrt(np.sum((x0-y0[0])**2*u0*dx)) 
y0[2:] = np.sqrt(y0[1]*gbm_pdf(y0[1]*z+y0[0],t0,loc,p[0],p[1]))

tspan=[t0, tf]
p1 = (z, Dz, Dz2, Mz, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
res = ivp_solver(fun_wave, Jac_wave, tspan, y0, pfun=p1, pjac=p2, newton_tol=1e-10)

mf = res['y'][0]
sf = res['y'][1]
bf = res['y'][2:]
xf = sf*z+mf
uf = bf*bf/sf

error = np.max(np.abs(uf-gbm_pdf(xf,tf,loc,p[0],p[1])))
print(f"max error = {error}")

x = np.linspace(np.min(xf),np.max(xf),1000)
uf_true = gbm_pdf(x,tf,loc,p[0],p[1])

plt.figure()
plt.plot(x,uf_true,label="Analytical")
plt.plot(xf,uf,".",label="Numerical")
plt.xlabel("x: space")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%



from scipy.optimize import approx_fprime

def compute_jacobian(fun, t, y, z, Dz, Dz2, M, a, D, p):
    """
    Computes the Jacobian of the function `fun` with respect to y using finite differences.
    """
    
    epsilon=1e-8
    
    
    def wrapped_fun(y_flat):
        y_vec = y_flat.reshape(y.shape)
        return fun(t, y_vec, z, Dz, Dz2, M, a, D, p).flatten()
    
    jacobian = approx_fprime(y.flatten(), wrapped_fun, epsilon)
    return jacobian.reshape(y.size, y.size)

J = Jac_wave(res['t'], res['y'], *p2)
J_test = compute_jacobian(fun_wave, res['t'], res['y'], *p1)
err_jac = J - J_test
print(f"jacobian error = {np.max(np.abs(err_jac))}")

eigs = np.linalg.eigvals(J)
Re = np.real(eigs)
Im = np.imag(eigs)

eigs_test = np.linalg.eigvals(J_test)
Re_test = np.real(eigs_test)
Im_test = np.imag(eigs_test)

plt.figure()
plt.plot(Re,Im,".",label="analytical")
plt.plot(Re_test,Im_test,".",label="finite difference")
plt.grid(True)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.show()


