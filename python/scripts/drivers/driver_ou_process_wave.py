import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander
from src.ivp_solver import ivp_solver, fun_wave, Jac_wave
from scripts.systems.ou_process import a,D,dadx,dDdx
from scipy.stats import norm

"""
Ornstein–Uhlenbeck process
dXt = p1(p2-Xt)*dt+p3*dBt
Diffusion: D = p3**2/2
"""

N = 16
z,w = nodes(N,Prob=True)
V,Vz = vander(z,Prob=True)
Vinv = np.linalg.inv(V)
Dz = Vz @ Vinv
Dz2 = Dz@Dz
Mz = Vinv.T @ Vinv
Mzd = np.diag(Mz) 

t0 = 0
tf = 20.5
p = np.array([0.3,5.0,0.5])


loc = 10
x0 = np.linspace(loc-5,loc+5,1000)
u0 = norm.pdf(x0,loc,scale=1)
dx = x0[1] - x0[0]
y0 = np.zeros(N+2)
y0[0]  = np.sum(x0*u0*dx) 
y0[1]  = np.sqrt(np.sum((x0-y0[0])**2*u0*dx)) 
y0[2:] = np.sqrt(y0[1]*norm.pdf(y0[1]*z+y0[0],loc,scale=1))


tspan=[t0, tf]
p1 = (z, Dz, Mzd, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
res = ivp_solver(fun_wave, Jac_wave, tspan, y0, pfun=p1, pjac=p2, newton_tol=1e-12)

mf = res['y'][0]
sf = res['y'][1]
bf = res['y'][2:]
xf = sf*z+mf
uf = bf*bf/sf

x = np.linspace(np.min(xf),np.max(xf),100)
plt.figure()
plt.plot(x0,u0,label=r"$u(x,t_0)$")
plt.plot(xf,uf,".-",label=r"$u(x,t_f)$")
plt.xlabel("x: space")
plt.title("Initial condition and stationary distribution")
plt.legend()
plt.tight_layout()
plt.show()

#%%

from scipy.optimize import approx_fprime

def compute_jacobian(fun, t, y, p1):
    """
    Computes the Jacobian of the function `fun` with respect to y using finite differences.
    """
    
    epsilon=1e-12
    
    
    def wrapped_fun(y_flat):
        y_vec = y_flat.reshape(y.shape)
        return fun(t, y_vec, *p1).flatten()
    
    jacobian = approx_fprime(y.flatten(), wrapped_fun, epsilon)
    return jacobian.reshape(y.size, y.size)

J = Jac_wave(res['t'], res['y'], *p2)
J_test = compute_jacobian(fun_wave, res['t'], res['y'], p1)
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


