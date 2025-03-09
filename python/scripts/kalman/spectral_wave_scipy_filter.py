import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Load data
"""

kalman_data_path = "/home/max/Documents/DTU/MasterThesis/RSDE/data/kalman_data.csv"
data = pd.read_csv(kalman_data_path)
tm = data.tm.to_numpy()
ym = data.ym.to_numpy()

"""
Define likelihood
"""

def likelihood(x,y,C,D):
    return np.exp(-(y-C*x)**2/(2*D**2))/np.sqrt(2*np.pi*D**2)


"""
Setup Initial Conditions and Parameters
"""

from src.hermite import nodes,vander
from src.ivp_solver import fun_wave
from scripts.systems.ou_process import a,D,dadx,dDdx
from scipy.integrate import solve_ivp

"""
Ornstein–Uhlenbeck process
dXt = p1(p2-Xt)*dt+p3*dBt
Diffusion: D = p3**2/2
"""

# Initialize grid and matrices
N = 32
z,w = nodes(N,Prob=True)

# Matrices based on Hermite Functions
V,Vz = vander(z,Prob=True)
Vinv = np.linalg.inv(V)
Mz = (Vinv.T @ Vinv).T
Dz = Vz @ Vinv
Dz2 = Dz@Dz

# initial condition
initial_condition = np.zeros(N+2)
initial_condition[1] = np.sqrt(2.0)
bhat = np.zeros(N)
bhat[0] = 1
initial_condition[2:] = V@bhat
state = initial_condition.copy()

#%%

kalman_data_parameters_path = "/home/max/Documents/DTU/MasterThesis/RSDE/data/kalman_data_parameters.csv"
p_data = pd.read_csv(kalman_data_parameters_path)

# paramteres (Uffes bog side 246)
Ap = p_data.A[0]
Gp = p_data.G[0]
Cp = p_data.C[0]
Dp = p_data.D[0]

tnow = 0.0

# setup parameters
p = np.array([np.abs(Ap),0.0,Gp])
p1 = (z, Dz, Dz2, Mz, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)

# define results array
res = np.zeros([len(tm),2+N])

for idx, measurement in enumerate(data.itertuples(index=False)):
    
    
    tnxt = measurement.tm
    ynxt = measurement.ym
    
    """
    Perform time update
    """
    # Define time interval
    tspan = [tnow,tnxt]
    
    # Solve ODE system
    sol = solve_ivp(fun_wave, tspan, state, args=(p1))
    state = sol['y'][:,-1]
    
    # Extract time
    tnow = tnxt

    """
    Perform data update
    """
    
    # Extract mean, standard deviation and solution
    m = state[0]
    s = state[1]
    b = state[2:]
    w = b*b
    
    # Construct grids
    x = s*z+m

    # Compute State Likelihood
    like = likelihood(x,ynxt,Cp,Dp)

    # Compute posterior using Bayes rule
    u = like*(w/s)
    u = u/(s*np.sum(Mz@u,axis=0))

    # Find new mean and standard deviation
    m = s*(x@Mz@u)
    s = np.sqrt(s*((x-m)**2@Mz@u))

    # Compute new z grids
    znew = (x-m)/s

    # Construct Vandermonde matrices for interpolation
    Vnew,_ = vander(znew,Prob=True)
    Vnewinv = np.linalg.inv(Vnew)
    
    # Compute w on original z grid via interpolation
    b = V @ Vnewinv @ np.sqrt(s*u)
    
    # update state vectors
    state[0]  = m
    state[1]  = s
    state[2:] = b

    # update result array
    res[idx] = state
    
    print(idx)
    

#%%

"""
Plot actual realization, data, and "mean reconstruction"
"""

kalman_data_sim_path = "/home/max/Documents/DTU/MasterThesis/RSDE/data/kalman_data_sim_path.csv"
sim_data = pd.read_csv(kalman_data_sim_path)

t = sim_data.t.to_numpy()
X = sim_data.X.to_numpy()

plt.figure()
plt.plot(t,X)
plt.plot(tm,ym,".")
plt.plot(tm,res[:,0],color="red")
plt.xlabel("t: time")
plt.ylabel(r"$X_t$: state")
plt.grid(True)
plt.title(f"Spectral Method with N = {N}")
plt.show()
    
#%%

# x_arr = res[:,1]*z[:,np.newaxis] + res[:,0]

# plt.figure()
# plt.plot(tm,x_arr.T,".",color="black",markersize=2)
# plt.xlabel("t: time")
# plt.xlabel("x: space")
# plt.title("Movement of grid points")
# plt.show()
    
    
    
    