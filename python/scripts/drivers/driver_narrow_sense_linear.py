import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander
from src.ivp_solver import ivp_solver, fun, Jac
from scripts.systems.narrow_sense_linear import a,D,dadx,dDdx
from scipy.stats import norm


N = 32
z,w = nodes(N)
V,Vz = vander(z)

Vinv = np.linalg.inv(V)

Dz = Vz @ Vinv
Dz2 = Dz@Dz
Mz = (Vinv.T @ Vinv).T

t0 = 0
tf = 20.5
p = np.array([0.3,1.0])


x0 = np.linspace(-10,10,1000)
u0 = norm.pdf(x0,loc=0,scale=1)
dx = x0[1] - x0[0]
y0 = np.zeros(N+2)
y0[0]  = np.sum(x0*u0*dx) 
y0[1]  = np.sqrt(np.sum((x0-y0[0])**2*u0*dx)) 
y0[2:] = y0[1]*norm.pdf(y0[1]*z+y0[0],loc=0,scale=1)


tspan=[t0, tf]
p1 = (z, Dz, Dz2, Mz, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
res = ivp_solver(fun, Jac, tspan, y0, pfun=p1, pjac=p2, newton_tol=1e-12)

mf = res['y'][0]
sf = res['y'][1]
wf = res['y'][2:]
xf = sf*z+mf
uf = wf/sf

x = np.linspace(np.min(xf),np.max(xf),100)
plt.figure()
plt.plot(xf,uf,".-",label=r"$u(x,t_f)$")
plt.xlabel("x: space")
plt.title("Initial condition and stationary distribution")
plt.legend()
plt.tight_layout()
plt.show()




