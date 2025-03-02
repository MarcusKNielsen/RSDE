import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander
from src.ivp_solver import ivp_solver, fun_wave, Jac_wave
from scripts.systems.cir import a,D,dadx,dDdx

"""
Cox–Ingersoll–Ross model
dXt = p1*(p2-Xt)*dt+p3*Xt*dBt
"""
from scipy.special import iv

def cir_pdf(x,t,x0,p):
    
    p1,p2,p3 = p
    
    c = 4 * p1 / (p3**2 * (1 - np.exp(-p1 * t)))
    q = 4 * p1 * p2 / p3**2
    x_bar = x0 * np.exp(-p1 * t) + p2 * (1 - np.exp(-p1 * t))
    
    pdf = (c / 2) * ((x / x_bar)**((q - 1) / 2)) * np.exp(-c * (x + x_bar) / 2) * iv(q-1, c * np.sqrt(x * x_bar))
    return pdf

N = 50
z,w = nodes(N,Prob=True)
V,Vz = vander(z,Prob=True)
Vinv = np.linalg.inv(V)
Dz = Vz @ Vinv
Dz2 = Dz@Dz
Mz = (Vinv.T @ Vinv).T

t0 = 0.2
tf = t0 + 1e-16
p = np.array([2.0,2.5,0.4])

loc = 2.0
x0 = np.linspace(1.0, 6, 1000)
u0 = cir_pdf(x0,t0,loc,p)
dx = x0[1] - x0[0]
y0 = np.zeros(N+2)
y0[0]  = np.sum(x0*u0*dx) 
y0[1]  = np.sqrt(np.sum((x0-y0[0])**2*u0*dx))
z0 = y0[1]*z+y0[0]
y0[2:] = np.sqrt(y0[1]*cir_pdf(y0[1]*z+y0[0],t0,loc,p))

plt.plot(x0,u0)
plt.plot(z0,np.zeros_like(z0),".")

#%%

tspan=[t0, tf]
p1 = (z, Dz, Dz2, Mz, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
res = ivp_solver(fun_wave, Jac_wave, tspan, y0, pfun=p1, pjac=p2, newton_tol=1e-10)

mf = res['y'][0]
sf = res['y'][1]
bf = res['y'][2:]
xf = sf*z+mf
uf = bf*bf/sf

error = np.max(np.abs(uf-cir_pdf(xf,tf,loc,p)))
print(f"max error = {error}")

x = np.linspace(np.min(xf),np.max(xf),1000)
uf_true = cir_pdf(x,tf,loc,p)

plt.figure()
plt.plot(x,uf_true,label="Analytical")
plt.plot(xf,uf,".",label="Numerical")
plt.xlabel("x: space")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
