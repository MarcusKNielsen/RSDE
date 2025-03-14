import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""
Load data
"""

# Define file path
path = "/home/max/Documents/DTU/MasterThesis/RSDE/data"
filename = "match_data.csv"

# Combine path and filename
file_path = f"{path}/{filename}"

# Read CSV into a DataFrame
data = pd.read_csv(file_path, dtype={
    "sim_idx": np.int32,
    "time": np.float64,
    "player1": np.int8,
    "player2": np.int8,
    "player1won": np.int8
})

#data = data[:50]

"""
Define likelihood
"""

def likelihood(x1,x2,y):
    
    X1,X2 = np.meshgrid(x1,x2)
    
    # win-loss-draw likelihood
    A = np.array([[1, -1], [0.0, -0.0], [-1, 1]])
    
    X_stack = np.stack([X1, X2], axis=-1)
    Ax = np.einsum('kj, nmj -> knm', A, X_stack)
    P = np.exp(Ax)
    Z = np.sum(P, axis=0)
    P = P / Z
    
    return P[1-y]

"""
Setup Filter
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
N = 16
z,w = nodes(N,Prob=True)

# Matrices based on Hermite Functions
V,Vz = vander(z,Prob=True)
Vinv = np.linalg.inv(V)
Mz = Vinv.T @ Vinv
Mzd = np.diag(Mz) 
Dz = Vz @ Vinv

# setup parameters
p = np.array([2.0,0.0,np.sqrt(2)])
p1 = (z, Dz, Mzd, a, D, p)
p2 = (z, Dz, Mz, a, D, dadx, dDdx, p)

# initial condition
initial_condition = np.zeros(N+2)
initial_condition[1] = np.sqrt(2)
bhat = np.zeros(N)
bhat[0] = 1
initial_condition[2:] = V@bhat


"""
Setup Player info
"""
from scripts.player_info import PlayerInfo

player_info = PlayerInfo(data, N+2)

"""
Run filter / Loop through data
"""

for match_idx, match_ in enumerate(data.itertuples(index=False)):
    
    print(match_idx)
    
    """
    Extract players data
    """
    match_num_player1 = player_info.players[match_.player1].matches_played
    match_num_player2 = player_info.players[match_.player2].matches_played
    
    if match_num_player1 == 0:
        y1 = initial_condition.copy()
    else:
        t1 = player_info.players[match_.player1].times[match_num_player1-1]
        y1 = player_info.players[match_.player1].data_matrix[match_num_player1-1]
    
    if match_num_player2 == 0:
        y2 = initial_condition.copy()
    else:
        t2 = player_info.players[match_.player2].times[match_num_player2-1]
        y2 = player_info.players[match_.player2].data_matrix[match_num_player2-1]

    """
    Perform time update
    """
    tf = match_.time
    
    if match_num_player1 != 0:
        tspan=[t1, tf]      
        res1 = solve_ivp(fun_wave, tspan, y1, args=(p1))
        y1 = res1['y'][:,-1]
    
    if match_num_player2 != 0:
        tspan=[t2, tf]
        res2 = solve_ivp(fun_wave, tspan, y2, args=(p1))
        y2 = res2['y'][:,-1]
    
    """
    Perform data update
    """
    # Extract mean, standard deviation and wave function
    m1 = y1[0]
    s1 = y1[1]
    b1 = y1[2:]
    w1 = b1*b1

    m2 = y2[0]
    s2 = y2[1]
    b2 = y1[2:]
    w2 = b2*b2
    
    # Construct grids
    x1 = s1*z+m1
    x2 = s2*z+m2
    
    # Extract outcome of match
    outcome = match_.player1won
    
    # Compute State Likelihood
    like = likelihood(x1,x2,outcome)
    
    # Compute marginal state likelihoods
    like1 = (w2*Mzd).T @ like   # dx2 integral
    like2 = (w1*Mzd).T @ like.T # dx1 integral
    
    # Compute posterior using Bayes rule
    u1 = like1*(w1/s1)
    u1 = u1/(s1*np.sum(Mzd*u1))

    u2 = like2*(w2/s2)
    u2 = u2/(s2*np.sum(Mzd*u2))

    # Pre compute "half" integral for reuse
    Mu1 = Mzd*u1
    Mu2 = Mzd*u2

    # Find new mean and standard deviation
    m1 = s1*(x1@Mu1)
    s1 = np.sqrt(s1*((x1-m1)**2@Mu1))
    
    m2 = s2*(x2@Mu2)
    s2 = np.sqrt(s2*((x2-m2)**2@Mu2))
    
    # Compute new z grids
    z1 = (x1-m1)/s1
    z2 = (x2-m2)/s2
    
    # Construct Vandermonde matrices for interpolation
    V1,_ = vander(z1,Prob=True)
    V2,_ = vander(z2,Prob=True)
    
    V1inv = np.linalg.inv(V1)
    V2inv = np.linalg.inv(V2)
    
    # Compute wave functions on original z grid via interpolation
    b1 = V @ (V1inv @ np.sqrt(s1*u1))
    b2 = V @ (V2inv @ np.sqrt(s2*u2))
    
    # update state vectors
    y1[0]  = m1
    y1[1]  = s1
    y1[2:] = b1

    y2[0]  = m2
    y2[1]  = s2
    y2[2:] = b2
    
    player_info.players[match_.player1].data_matrix[match_num_player1] = y1
    player_info.players[match_.player2].data_matrix[match_num_player2] = y2

    player_info.players[match_.player1].matches_played += 1
    player_info.players[match_.player2].matches_played += 1


#%%


def get_density_of_player(player,player_info,x_large):

    M = len(x_large)    

    t = player_info.players[player].times
    m = player_info.players[player].data_matrix[:,0]
    s = player_info.players[player].data_matrix[:,1]
    b = player_info.players[player].data_matrix[:,2:]
    w = b*b
    
    z_large = (x_large - m[:,np.newaxis])/s[:,np.newaxis]
    w_large = np.zeros([len(t),M])
    
    for i in range(len(t)):
        V_large,_ = vander(z_large[i],N)
        w_large[i] = V_large@Vinv@w[i]
    
    u_large = w_large/s[:, np.newaxis]

    return t,u_large    


# Read CSV into a DataFrame
sim_data = pd.read_csv(f"{path}/sim_data.csv")
N_players = sim_data.shape[1] - 1
x_large = np.linspace(-5,5,100)

for player in range(N_players):
    t,u_large = get_density_of_player(player,player_info,x_large)
    T,X = np.meshgrid(t,x_large)
    plt.figure()
    plt.pcolormesh(T,X,u_large.T)
    plt.plot(sim_data.time,sim_data.iloc[:,player+1],color="red")
    plt.xlabel("t: time")
    plt.ylabel("x: space")
    plt.title(f"Player = {player}")
    plt.xlim([t[0],t[-1]])
    plt.tight_layout()
    plt.show()













