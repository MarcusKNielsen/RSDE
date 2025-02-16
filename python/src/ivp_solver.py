import numpy as np
from numpy.linalg import norm
from numpy import size, identity
from scipy.linalg import lu, solve_triangular

"""
runge kutta update is based on the esdirk23 method
"""
def runge_kutta_update(fun, jac, xn, tn, h, k3, pfun, pjac, newton_tol, newton_maxiter, res):
    
    gamma = (2 - np.sqrt(2)) / 2
    
    c2 = 2 * gamma
    c3 = 1

    a21 = gamma
    a22 = gamma
    
    a31 = (1 - gamma) / 2
    a32 = (1 - gamma) / 2
    a33 = gamma
    
    d1 = (1 - 6 * gamma**2) / (12 * gamma)
    d2 = (6 * gamma * (1 - 2 * gamma) * (1 - gamma) - 1) / (12 * gamma * (1 - 2 * gamma))
    d3 = (6 * gamma * (1 - gamma) - 1) / (3 * (1 - 2 * gamma))

    T2 = tn + c2 * h
    T3 = tn + c3 * h

    k1 = k3

    xinit = xn + c2 * h * k1
    
    dt = a22 * h
    c  = xn + a21 * h * k1

    X2 = newtons_method(fun, jac, T2, dt, xinit, c, newton_tol, newton_tol, pfun, pjac, res)
    k2 = fun(T2, X2, *pfun)
    
    dt = a33 * h
    c  = xn + a31 * h * k1 + a32 * h * k2

    X3 = newtons_method(fun, jac, T3, dt, xinit, c, newton_tol, newton_tol, pfun, pjac, res)
    k3 = fun(T3, X3, *pfun)
    
    xnxt = X3
    enxt = h * (d1 * k1 + d2 * k2 + d3 * k3)
    
    res['nfev'] += 2
    
    return xnxt, enxt, k3

"""
inexact newton solver
"""
def newtons_method(fun, jac, tnxt, dt, xinit, c, tol, maxiter, pfun, pjac, res):
    
    x = xinit  
    k = 0  

    I = identity(size(x))
    
    Rnow = x - dt * fun(tnxt, x, *pfun) - c
    a = 1
    res['nfev'] += 1


    while (k < maxiter and norm(Rnow) > tol):
        
        k += 1
        
        if a >= 1:
            JR = I - dt * jac(tnxt, x, *pjac)
            P, L, U = lu(JR)
            res['njev'] += 1
            res['nluf'] += 1
            
        b = P @ (-Rnow)
        y = solve_triangular(L, b, lower=True)
        dx = solve_triangular(U, y, lower=False)
        res['nlub'] += 1
        
        x = x + dx
        Rold = Rnow
        Rnow = x - dt * fun(tnxt, x, *pfun) - c
        a = norm(Rnow) / norm(Rold)
        res['nfev'] += 1

    return x

"""
initial value problem solver

dy/dt = f(t,y,p)
y0 = y(t0)

"""
def ivp_solver(fun, jac, tspan, y0, **options):

    abstol = options.get("abstol", 1e-6)
    reltol = options.get("reltol", 1e-3)
    epstol = options.get("epstol", 0.9)
    min_step = options.get("min_step", 0.0)
    max_step = options.get("min_step", np.inf)
    pfun = options.get("pfun", ())
    pjac = options.get("pjac", ())
    newton_tol = options.get("newton_tol", 1e-6)
    newton_maxiter = options.get("newton_maxiter", 25)

    res = dict()
    res['nfev'] = 0
    res['njev'] = 0
    res['nluf'] = 0
    res['nlub'] = 0

    t0 = tspan[0]
    tf = tspan[1]
    dt = 1e-6
    y  = y0
    t  = t0

    k3 = fun(t0, y0, *pfun)  
    
    res['nfev'] += 1
    
    while t < tf:
        
        if t + dt > tf:
            dt = tf - t

        AcceptStep = False
        while not AcceptStep:
            
            ynxt, err, k3 = runge_kutta_update(fun, jac, y, t, dt, k3, pfun, pjac, newton_tol, newton_maxiter, res)
            
            r = np.max(np.abs(err) / np.maximum(abstol, np.abs(ynxt) * reltol))
            
            AcceptStep = (r <= 1)
            if AcceptStep:
                t = t + dt
                y = ynxt
                
            dt = max(min_step, min((epstol/r)**(1/3), max_step)) * dt
    
    res['t'] = t
    res['y'] = y
    
    return res

"""
right hand side funtion for advection-diffusion system on smart form:

dy/dt = f(t,y,p)    

"""

def fun(t,y,z,Dz,Dz2,M,a,D,p):
    
    mu = y[0]
    s  = y[1]
    w  = y[2:]
    
    f = np.zeros_like(y)
    x = s*z+mu
    e = np.ones_like(z)
    
    Dzw = Dz@w
    F  = a(t,x,p)*w - (D(t,x,p)/s)*(Dzw)
    MF = M@F
    R  = e@MF
    Q  = z@MF
    
    f[0]  = R
    f[1]  = Q
    #f[2:] = ((Q*z+R)*Dzw + Q*w - (Dz@F))/s
    f[2:] = Dz@(((Q*z+R)*w - F)/s)
    #f[2:] = -Dz@((a(t,x,p)-Q*z-R)*w/s - (D(t,x,p)/s**2)*Dzw)
    
    return f

"""
jac of right hand side funtion f
"""

def Jac(t, y, z, Dz, M, a, D, dadx, dDdx, p):
    
    mu = y[0]
    s  = y[1]
    w  = y[2:]
    
    x = s*z+mu
    e = np.ones_like(z)

    Dzw = Dz @ w
    F  = a(t,x,p) * w - (D(t,x,p) / s) * Dzw
    eM = e @ M
    zM = z @ M
    R  = eM @ F
    Q  = zM @ F
    
    # Compute derivatives
    dFdmu = dadx(t,x,p) * w - (1/s) * dDdx(t,x,p) * Dzw
    dRdmu = np.array([[eM @ dFdmu]])  # (1,1)
    dQdmu = np.array([[zM @ dFdmu]])  # (1,1)
    
    dSdmu = (1/s) * ((dQdmu * z + dRdmu * e) * Dzw + dQdmu * w - Dz @ dFdmu)
    dSdmu = dSdmu.reshape(len(z), 1)  # (N,1)

    dFds = dadx(t,x,p) * z * w - (-D(t,x,p) / s**2 + (1/s) * dDdx(t,x,p) * z) * Dzw
    dRds = np.array([[eM @ dFds]])  # (1,1)
    dQds = np.array([[zM @ dFds]])  # (1,1)
    
    dSds = (-1/s**2) * ((Q * z + R * e) * Dzw + Q * w - Dz @ F) + (1/s) * ((dQds * z + dRds * e) * Dzw + dQds * w - Dz @ dFds)
    dSds = dSds.reshape(len(z), 1)  # (N,1)

    dFdw = np.diag(a(t,x,p)) - (1/s) * (np.diag(D(t,x,p)) @ Dz)
    dRdw = eM @ dFdw  # (1,N)
    dQdw = zM @ dFdw  # (1,N)

    wdQdw = np.outer(w, dQdw)  # (N,N)
    wdRdw = np.outer(w, dRdw)  # (N,N)
    
    RI = np.diag(R * e)  # (N,N)
    QI = np.diag(Q * e)  # (N,N)
    Z = np.diag(z)  # (N,N)

    dSdw = (1/s) * (Z @ Dz @ (QI + wdQdw) + Dz @ (RI + wdRdw) + QI + wdQdw - Dz @ dFdw)  # (N,N)
    
    # block structure
    J = np.block([
        [dRdmu, dRds, dRdw],  # (1,1), (1,1), (1,N)
        [dQdmu, dQds, dQdw],  # (1,1), (1,1), (1,N)
        [dSdmu, dSds, dSdw]   # (N,1), (N,1), (N,N)
    ])
    
    return J

