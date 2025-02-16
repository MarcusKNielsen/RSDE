import numpy as np

"""
Population Model
dXt = p1*Xt*(p2-Xt)*dt+p3*Xt*dBt
Diffusion: D(t,x) = (p3*x)**2/2
"""

# drift
def f(t,x,p):
    p1,p2,p3 = p
    return p1*x*(p2-x)

# noise intensity
def g(t,x,p):
    p1,p2,p3 = p
    return p3*x

# advection
def a(t,x,p):
    p1,p2,p3 = p
    return p1*x*(p2-x) - p3**2

# diffusion
def D(t,x,p):
    p1,p2,p3 = p
    return (p3*x)**2/2

def dadx(t,x,p):
    p1,p2,p3 = p
    return p1*(p2-2*x)

def dDdx(t,x,p):
    p1,p2,p3 = p
    return p3**2*x