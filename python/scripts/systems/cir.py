import numpy as np

"""
Cox–Ingersoll–Ross model
dXt = p1*(p2-Xt)*dt+p3*Xt*dBt
"""

# drift
def f(t,x,p):
    p1,p2,p3 = p
    return p1*(p2-x)

# noise intensity
def g(t,x,p):
    p1,p2,p3 = p
    return p3*np.sqrt(x)

# advection
def a(t,x,p):
    p1,p2,p3 = p
    return p1*(p2-x)-p3**2/2

# diffusion
def D(t,x,p):
    p1,p2,p3 = p
    return p3**2*x/2

def dadx(t,x,p):
    p1,p2,p3 = p
    return (-p1)*np.ones_like(x)

def dDdx(t,x,p):
    p1,p2,p3 = p
    return (p3**2/2)*np.ones_like(x)

