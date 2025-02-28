import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

"""
Setup right hand side of ODE system
"""

def fun(t,y,A,G):
    H = np.array([[A,0],
                  [0,2*A]])
    
    b = np.array([0,G**2])
    
    return H@y+b

"""
Load data
"""

kalman_data_path = "/home/max/Documents/DTU/MasterThesis/RSDE/data/kalman_data.csv"
data = pd.read_csv(kalman_data_path)
tm = data.tm.to_numpy()
ym = data.ym.to_numpy()

"""
Setup Initial Conditions and Parameters
"""

kalman_data_parameters_path = "/home/max/Documents/DTU/MasterThesis/RSDE/data/kalman_data_parameters.csv"
p_data = pd.read_csv(kalman_data_parameters_path)

# paramteres (Uffes bog side 246)
A = p_data.A[0]
G = p_data.G[0]
C = p_data.C[0]
D = p_data.D[0]

state = np.array([0.0,100.0])
tnow = 0.0

# define results array
res = np.zeros([len(tm),2])

for idx, measurement in enumerate(data.itertuples(index=False)):
    
    tnxt = measurement.tm
    ynxt = measurement.ym
    
    """
    Perform time update
    """
    # Define time interval
    tspan = [tnow,tnxt]
    
    # Solve ODE system
    sol = solve_ivp(fun, tspan, state, method='RK45', args=(A,G))
    
    # Extract time
    tnow = sol['t'][-1]
    
    # Get mean and covariance
    mean = sol['y'][0][-1]
    cov  = sol['y'][1][-1]

    """
    Perform data update
    """
    
    # Compute Kalman Gain
    K = cov*C/(C*cov*C+D*D)

    # update mean
    mean = mean + K*(ynxt - C*mean)
    
    #update covariance
    cov = cov - K*C*cov

    # update state vector
    state[0] = mean
    state[1] = cov

    # update result array
    res[idx] = state


#%%

"""
Extend results array for plotting
"""

Nt = 20

res_exd = np.zeros([Nt*len(tm)-Nt,3])

for idx,state in enumerate(res[:-1]):
    
    # Define time interval
    tnow = tm[idx]
    tnxt = tm[idx+1]
    tspan = [tnow,tnxt]
    teval = np.linspace(tnow,tnxt,Nt)
    
    
    # Solve ODE system
    sol = solve_ivp(fun, tspan, state, method='RK45', t_eval=teval, args=(A,G))

    res_exd[idx*Nt:idx*Nt+Nt,0]  = sol.t
    res_exd[idx*Nt:idx*Nt+Nt,1:] = (sol.y).T
    

#%%

"""
Plot actual realization, data, and "mean reconstruction"
"""

import matplotlib.pyplot as plt

kalman_data_sim_path = "/home/max/Documents/DTU/MasterThesis/RSDE/data/kalman_data_sim_path.csv"
sim_data = pd.read_csv(kalman_data_sim_path)

t = sim_data.t.to_numpy()
X = sim_data.X.to_numpy()

plt.figure()
plt.plot(t,X)
plt.plot(tm,ym,".")
plt.plot(tm,res[:,0],".")
plt.plot(res_exd[:,0],res_exd[:,1])
plt.xlabel("t: time")
plt.ylabel(r"$X_t$: state")
plt.grid(True)
plt.show()

#%%

"""
Plot the full evolution of density
"""
import scipy as sp


x_grid = np.linspace(-3,3,200)
t_grid = res_exd[:,0]

T_grid,X_grid = np.meshgrid(t_grid,x_grid)

U = np.zeros_like(X_grid)

for idx in range(len(t_grid)):
    U[:,idx] = sp.stats.norm.pdf(x_grid,res_exd[idx,1],np.sqrt(res_exd[idx,2]))


plt.figure(figsize=(12,4))
c = plt.pcolormesh(T_grid,X_grid,U)
plt.colorbar(c)
plt.plot(t,X,color="red",linewidth=1.0)
plt.xlim([tm[0],tm[-1]])
plt.xlabel("t: time")
plt.ylabel(r"$X_t$: state")
plt.tight_layout()
plt.show()










