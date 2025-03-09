import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander
from src.ivp_solver import ivp_solver, fun, Jac, fun_wave, Jac_wave
from scripts.systems.gmb import a,D,dadx,dDdx
from scipy.integrate import solve_ivp

"""
Geometric Brownian Motion
dXt = p1*Xt*dt+p2*Xt*dBt
"""

def gbm_pdf(x,t,x0,p1,p2):
    return np.exp(-(np.log(x) - np.log(x0) - (p1-0.5*p2**2)*t)**2/(2*p2**2*t))/(x*p2*np.sqrt(2*np.pi*t))

def init_cond(t0,loc,p,is_wave=False):
    x0 = np.linspace(4,8,200)
    u0 = gbm_pdf(x0,t0,loc,p[0],p[1])
    dx = x0[1] - x0[0]
    y0 = np.zeros(N+2)
    y0[0]  = np.sum(x0*u0*dx) 
    y0[1]  = np.sqrt(np.sum((x0-y0[0])**2*u0*dx))
    if is_wave == False:
        y0[2:] = y0[1]*gbm_pdf(y0[1]*z+y0[0],t0,loc,p[0],p[1])
    else:
        y0[2:] = np.sqrt(y0[1]*gbm_pdf(y0[1]*z+y0[0],t0,loc,p[0],p[1]))
        
    return y0

# Global parameters
t0 = 0.10
tf = t0 + 1e-15
p = np.array([2.1,0.1])
loc = 5
xm = np.linspace(4,8,100)

Nlist = np.arange(2,50,1)
errors = np.zeros([len(Nlist),2])

Nsigns = np.zeros([len(Nlist),2])

use_scipy = False
method = "RK45"
#method = "LSODA"
#method = "RK23"
#method = "DOP853"
#method = "Radau"

for idx,N in enumerate(Nlist):

    """
    Solve for w(z,t)
    """
    # Setup matrices based on physicist hermite functions
    Prob = False
    z,w = nodes(N,Prob=Prob)
    V,Vz = vander(z,Prob=Prob)
    Vinv = np.linalg.inv(V)
    Dz = Vz @ Vinv
    Dz2 = Dz@Dz
    Mz = (Vinv.T @ Vinv).T
    
    y0 = init_cond(t0,loc,p,is_wave=False)
    
    tspan=[t0, tf]
    p1 = (z, Dz, Dz2, Mz, a, D, p)
    p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
    
    if use_scipy == False:
        res = ivp_solver(fun, Jac, tspan, y0, pfun=p1, pjac=p2, newton_tol=1e-10)
        mf = res['y'][0]
        sf = res['y'][1]
        wf = res['y'][2:] 
    else:
        res = solve_ivp(fun, tspan, y0, args=(p1),method=method)
        mf = res.y[0,-1]
        sf = res.y[1,-1]
        wf = res.y[2:,-1]
    
    xf = sf*z+mf
    uf = wf/sf
    
    zm = (xm-mf)/sf
    Vm,_ = vander(zm,N,Prob=Prob)
    wm = Vm@Vinv@wf
    um = wm/sf
    
    error = np.max(np.abs(um-gbm_pdf(xm,tf,loc,p[0],p[1])))
    errors[idx,0] = error
    
    Nsigns[idx,0] = np.sum(um<0)
    
    """
    Solve for psi(z,t)
    """
    # Setup matrices based on probabilistic hermite functions
    Prob = True
    z,w = nodes(N,Prob=Prob)
    V,Vz = vander(z,Prob=Prob)
    Vinv = np.linalg.inv(V)
    Dz = Vz @ Vinv
    Dz2 = Dz@Dz
    Mz = (Vinv.T @ Vinv).T
    
    y0 = init_cond(t0,loc,p,is_wave=True)
    
    tspan=[t0, tf]
    p1 = (z, Dz, Dz2, Mz, a, D, p)
    p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
    
    if use_scipy == False:
        res = ivp_solver(fun_wave, Jac_wave, tspan, y0, pfun=p1, pjac=p2, newton_tol=1e-10)
        mf = res['y'][0]
        sf = res['y'][1]
        bf = res['y'][2:]
    else:
        res = solve_ivp(fun_wave, tspan, y0, args=(p1),method=method)
        mf = res.y[0,-1]
        sf = res.y[1,-1]
        bf = res.y[2:,-1]
    
    xf = sf*z+mf
    uf = bf*bf/sf

    zm = (xm-mf)/sf
    Vm,_ = vander(zm,N,Prob=Prob)
    bm = Vm@Vinv@bf
    um = bm**2/sf
    
    error = np.max(np.abs(um-gbm_pdf(xm,tf,loc,p[0],p[1])))
    errors[idx,1] = error

    Nsigns[idx,1] = np.sum(um<0)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# First plot (Convergence Test)
axes[0].semilogy(Nlist, errors[:, 0], ".-", label=r"Based on $w(z,t)$")
axes[0].semilogy(Nlist, errors[:, 1], ".-", label=r"Based on $\Psi(z,t)$")
axes[0].set_xlabel(r"Number of Nodes: $N$")
axes[0].set_ylabel(r"Error: $\max|u_N - u|$")
axes[0].set_title("Convergence Test")
axes[0].grid(True)
axes[0].legend()

# Second plot (Sign Test)
axes[1].plot(Nlist, Nsigns[:, 0], ".-", label=r"Based on $w(z,t)$")
axes[1].plot(Nlist, Nsigns[:, 1], ".-", label=r"Based on $\Psi(z,t)$")
axes[1].set_xlabel(r"Number of Nodes: $N$")
axes[1].set_ylabel("Number of Negative signs")
axes[1].set_title("Sign Test")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()

