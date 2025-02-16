import numpy as np

"""
Ornstein–Uhlenbeck process
dXt = p1(p2-Xt)*dt+p3*dBt
Diffusion: D = p3**2/2
"""

# drift
def f(t,x,p):
    p1,p2,p3 = p
    return p1*(p2-x)

# noise intensity
def g(t,x,p):
    p1,p2,p3 = p
    return p3*np.ones_like(x)

#advection
def a(t,x,p):
    p1,p2,p3 = p
    return p1*(p2-x)

#diffusion
def D(t,x,p):
    p1,p2,p3 = p
    return (p3**2/2)*np.ones_like(x)

def dadx(t,x,p):
    p1,p2,p3 = p
    return -p1*np.ones_like(x)

def dDdx(t,x,p):
    return np.zeros_like(x)