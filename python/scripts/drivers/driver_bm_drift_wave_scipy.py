import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander
from src.ivp_solver import fun_wave
from scripts.systems.bm_drift import a,D,dadx,dDdx
from scipy.integrate import solve_ivp

"""
Brownian Motion with Drift
dXt = p1*dt+p2*dBt
"""

def gauss(t,x,a,D,loc):
    return np.exp(-(x-a*t-loc)**2/(4*D*t))/np.sqrt(4*np.pi*D*t)

N = 4
z,w = nodes(N,Prob=True)
V,Vz = vander(z,Prob=True)

Vinv = np.linalg.inv(V)

Dz = Vz @ Vinv
Dz2 = Dz@Dz
Mz = (Vinv.T @ Vinv).T

t0 = 10.0
tf = 100.0
p = np.array([1.0,1.0])

# initial condition
x0 = np.linspace(-2*t0*p[1],2*t0*p[1],1000)
loc = -p[0]*t0
u0 = gauss(t0,x0,p[0],p[1]**2/2,loc)
dx = x0[1] - x0[0]
y0 = np.zeros(N+2)
y0[0]  = np.sum(x0*u0*dx) 
y0[1]  = np.sqrt(np.sum((x0-y0[0])**2*u0*dx)) 
y0[2:] = np.sqrt(y0[1]*gauss(t0,y0[1]*z+y0[0],p[0],p[1]**2/2,loc))

tspan=[t0, tf]
p1 = (z, Dz, Dz2, Mz, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
res = solve_ivp(fun_wave, tspan, y0, args=(p1))

mf = res.y[0,-1]
sf = res.y[1,-1]
bf = res.y[2:,-1]
xf = sf*z+mf
uf = bf*bf/sf

x = np.linspace(np.min(xf),np.max(xf),100)
plt.figure()
plt.plot(x0,u0,label=r"$u(x,t_0)$")
plt.plot(x,gauss(tf,x,p[0],p[1]**2/2,loc),label="analytical")
plt.plot(xf,uf,".",label=r"$u(x,t_f)$")
plt.xlabel("x: space")
plt.legend()
plt.tight_layout()
plt.show()

error = gauss(tf,xf,p[0],p[1]**2/2,loc) - uf
print(f"error = {np.max(np.abs(error))}")
plt.figure()
plt.plot(xf,error)
plt.show()









