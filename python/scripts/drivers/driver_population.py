import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander
from src.ivp_solver import ivp_solver, fun, Jac
from scripts.systems.population_model import a,D,dadx,dDdx
from scipy.stats import norm

"""
Population Model
dXt = p1*Xt*(p2-Xt)*dt+p3*Xt*dBt
Diffusion: D(t,x) = (p3*x)**2/2
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
tf = 1
p = np.array([0.3,5.0,0.1])


loc = 5
scale = 0.1
x0 = np.linspace(loc-6*scale,loc+6*scale,1000)
u0 = norm.pdf(x0,loc,scale)
dx = x0[1] - x0[0]
y0 = np.zeros(N+2)
y0[0]  = np.sum(x0*u0*dx) 
y0[1]  = np.sqrt(np.sum((x0-y0[0])**2*u0*dx)) 
y0[2:] = y0[1]*norm.pdf(y0[1]*z+y0[0],loc,scale)


tspan=[t0, tf]
p1 = (z, Dz, Mz, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
res = ivp_solver(fun, Jac, tspan, y0, pfun=p1, pjac=p2, newton_tol=1e-12)

mf = res['y'][0]
sf = res['y'][1]
wf = res['y'][2:]
xf = sf*z+mf
uf = wf/sf

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

J = Jac(res['t'], res['y'], *p2)
eigs = np.linalg.eigvals(J)
Re = np.real(eigs)
Im = np.imag(eigs)
plt.figure()
plt.plot(Re,Im,".")
#plt.plot(Re[Re>0],Im[Re>0],".",color="red")
plt.vlines(0, np.min(Im), np.max(Im), linestyle = "--", color="red")
plt.grid(True)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.show()

print(f"max real eig = {Re.max()}")


