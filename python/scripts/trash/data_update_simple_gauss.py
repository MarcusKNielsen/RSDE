import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Given data
x = np.linspace(-10, 10, 1000)  # Example x values
u = np.exp(-0.5 * (x) ** 2)  # Example u(x), a Gaussian-like function
dx = x[1] - x[0]
u = u/np.sum(u*dx)
mu = np.sum(x*u*dx)
su = np.sqrt(np.sum((x-mu)**2*u*dx))
print(f"mu = {mu}")
print(f"su = {su}")

# Define z values (you can choose these as needed)
z = np.linspace(-10, 10, 1000)  # Example z values

# Interpolation function for u(x)
u_interp = interp1d(x, u, kind='linear', fill_value='extrapolate')

# Compute transformed values
w = su * u_interp(su * z + mu)

# w is now the transformed array
dz = z[1] - z[0]
pw = np.sum(w*dz)
mw = np.sum(z*w*dz)
sw = np.sqrt(np.sum((z-mw)**2*w*dz))

plt.plot(z,w)
