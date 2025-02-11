import numpy as np

"""
Narrow Sense Linear SDE
dXt = p1*Xt*dt+p2*dBt
Diffusion: D = p2**2/2
"""

# drift
def f(t,x,p):
    p1,p2 = p
    return p1*x

# noise intensity
def g(t,x,p):
    p1,p2 = p
    return p2*np.ones_like(x)

#advection
def a(t,x,p):
    p1,p2 = p
    return p1*x

#diffusion
def D(t,x,p):
    p1,p2 = p
    return (p2**2/2)*np.ones_like(x)

def dadx(t,x,p):
    p1,p2 = p
    return p1*np.ones_like(x)

def dDdx(t,x,p):
    return np.zeros_like(x)