import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander
from src.ivp_solver import fun_wave
from scripts.systems.gmb import a,D,dadx,dDdx
from scipy.integrate import solve_ivp

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
Mz = Vinv.T @ Vinv
Mzd = np.diag(Mz) 

t0 = 0.1
tf = t0 + 1.0
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
p1 = (z, Dz, Mzd, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
res = solve_ivp(fun_wave, tspan, y0, args=(p1))

mf = res.y[0,-1]
sf = res.y[1,-1]
bf = res.y[2:,-1]
xf = sf*z+mf
uf = bf*bf/sf

x = np.linspace(np.min(xf),np.max(xf),200)
plt.figure()
#plt.plot(x0,u0,label=r"$u(x,t_0)$")
plt.plot(x,gbm_pdf(x,tf,loc,p[0],p[1]),label="analytical")
plt.plot(xf,uf,".",label=r"$u(x,t_f)$")
plt.xlabel("x: space")
plt.legend()
plt.tight_layout()
plt.show()

error = gbm_pdf(xf,tf,loc,p[0],p[1]) - uf
print(f"error = {np.max(np.abs(error))}")
plt.figure()
plt.plot(xf,error)
plt.show()



