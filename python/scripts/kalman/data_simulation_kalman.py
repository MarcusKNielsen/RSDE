from scripts.SDE_simulation import simulate_sde
import numpy as np
import pandas as pd

# Example usage:
num_of_simulations = 1
t0 = 0.0
tf = 10.0
dt = 0.001
X0 = 0

# Parameters
A = -2
G = np.sqrt(2)
C = 1
D = 0.5

def f(t, X):
    return A*X

def g(t, X):
    return G


# Fix Randomness
seed = 1

t, X = simulate_sde(num_of_simulations, tf, dt, X0, f, g, seed)
X = X[:,0]

tm = t[::100]
xm = X[::100]

N = len(tm)

# Get random numbers
Z = np.random.normal(loc=0, scale=1, size=N)  # loc=mean, scale=std deviation

ym = C*xm + D*Z


data = pd.DataFrame({
    "tm": tm,
    "ym": ym
})

sim_data = pd.DataFrame({
    "t": t,
    "X": X
})

parameters_data = pd.DataFrame({
    "A": np.array([A]),
    "G": np.array([G]),
    "C": np.array([C]),
    "D": np.array([D]),
})

# Desktop
kalman_data_path = "/home/max/Documents/DTU/MasterThesis/RSDE/data/kalman_data.csv"
kalman_data_sim_path = "/home/max/Documents/DTU/MasterThesis/RSDE/data/kalman_data_sim_path.csv"
kalman_data_parameters_path = "/home/max/Documents/DTU/MasterThesis/RSDE/data/kalman_data_parameters.csv"

data.to_csv(kalman_data_path, index=False)
sim_data.to_csv(kalman_data_sim_path, index=False)
parameters_data.to_csv(kalman_data_parameters_path, index=False)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(t,X)
plt.plot(tm,ym,".")
plt.xlabel("t: time")
plt.ylabel(r"$X_t$: state")
plt.show()




