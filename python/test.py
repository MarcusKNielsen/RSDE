import numpy as np
import matplotlib.pyplot as plt
from src.hermite import nodes,vander
from scipy.integrate import solve_ivp
from src.ivp_solver import ivp_solver, fun, Jac
from scripts.systems.bm_drift import a,D,dadx,dDdx

def skew_normal_pdf(x, alpha, mu=0):
    from scipy.special import erf
    phi_x = (1 / np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / 2)
    Phi_alpha_x = 0.5 * (1 + erf(alpha * (x - mu) / np.sqrt(2)))
    return 2 * phi_x * Phi_alpha_x

N = 24
z,w = nodes(N)
V,Vz = vander(z)

Vinv = np.linalg.inv(V)

Dz = Vz @ Vinv
Dz2 = Dz@Dz
Mz = (Vinv.T @ Vinv).T

t0 = 0.0
tf = 100.1
p = np.array([-0.5,0.2])

# initial condition (skew-gauss)
skew = 0
x0 = np.linspace(-10,10,1000)
u0 = skew_normal_pdf(x0, skew)
dx = x0[1] - x0[0]
y0 = np.zeros(N+2)
y0[0]  = np.sum(x0*u0*dx) 
y0[1]  = np.sqrt(np.sum((x0-y0[0])**2*u0*dx)) 
y0[2:] = y0[1]*skew_normal_pdf(y0[1]*z+y0[0], skew)

max_step = np.inf
res = solve_ivp(fun,[t0,tf],y0,args=(z,Dz,Dz2,Mz,a,D,p),method="Radau",max_step=max_step)

res1 = solve_ivp(
    fun, 
    [t0, tf], 
    y0, 
    args=(z,Dz,Dz2,Mz,a,D,p),
    method="Radau", 
    max_step=max_step,
    jac=lambda t, y, *args: Jac(t,y,z,Dz,Mz,a,D,dadx,dDdx,p)
)

T = res.t

plt.figure()
mf = res.y[0,-1]
sf = res.y[1,-1]
wf = res.y[2:,-1]
xf = sf*z+mf
uf = wf/sf

mf1 = res1.y[0,-1]
sf1 = res1.y[1,-1]
wf1 = res1.y[2:,-1]
xf1 = sf1*z+mf1
uf1 = wf1/sf1

plt.plot(x0,u0,label=r"$u(x,t_0)$")
plt.plot(xf,uf,".-",label=r"$u(x,t_f)$")
plt.plot(xf1,uf1,".-",label=r"$u(x,t_f)$ with jac")

# J = Jac(t0,y0,z,Dz,M,a,D,dadx,dDdx,p)
# eigs = np.linalg.eigvals(J)
# plt.plot(np.real(eigs),np.imag(eigs),".")

tspan=[t0, tf]
p1 = (z, Dz, Dz2, Mz, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)
res2 = ivp_solver(fun, Jac, tspan, y0, pfun=p1, pjac=p2)

mf = res2['y'][0]
sf = res2['y'][1]
wf = res2['y'][2:]
xf = sf*z+mf
uf = wf/sf

plt.plot(xf,uf,".-",label=r"$u(x,t_f)$ with ESDIRK23")

plt.xlabel("x: space")
plt.title("Initial condition and stationary distribution")
plt.legend()
plt.tight_layout()
plt.show()



