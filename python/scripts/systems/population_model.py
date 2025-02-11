import numpy as np

"""
Geometric Brownian Motion
dXt = p1*Xt*(p2-Xt)*dt+p3*Xt*dBt
Diffusion: D(t,x) = (p2*x)**2/2
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
    return 

# diffusion
def D(t,x,p):
    p1,p2,p3 = p
    return 

def dadx(t,x,p):
    p1,p2,p3 = p
    return 

def dDdx(t,x,p):
    p1,p2,p3 = p
    return 