import numpy as np
from scripts.SDE_simulation import simulate_sde
import pandas as pd


def gen_data(num_of_players,num_of_matches,possible_outcomes,seed,p_sde_sim,like,p_like):

    np.random.seed(seed)    

    # Simulate SDE paths
    t, S = simulate_sde(num_of_players,*p_sde_sim)
    
    # Get random indexes for matches
    sim_idx = np.random.choice(np.arange(t.size), size=num_of_matches, replace=False)
    sim_idx = np.sort(sim_idx)
    
    match_up = np.array([np.random.choice(np.arange(num_of_players), size=2, replace=False) for _ in range(num_of_matches)])
    
    outcomes = np.zeros(num_of_matches)
    
    for match_num in range(num_of_matches):
    
        # get match indexes in S
        match_idx = sim_idx[match_num]
        
        # Get rating of players
        x = S[match_idx][match_up[match_num]]
        
        # Get probability distribution of outcomes
        P = like(x,*p_like)
        
        # Draw random outcome
        outcomes[match_num] = np.random.choice(possible_outcomes, p=P)
    
    """
    Save data in pandas dataframe
    """
    import os
    
    sim_data = pd.DataFrame(np.hstack((t.reshape(-1, 1), S)), columns=["time"] + [f"Player{i}" for i in range(num_of_players)])
    
    
    match_data = pd.DataFrame({
        "sim_idx": sim_idx,
        "time": t[sim_idx],
        "player1": match_up[:, 0],
        "player2": match_up[:, 1],
        "player1won": outcomes
    })
    
    # Convert types explicitly
    match_data = match_data.astype({
        "sim_idx": np.int32,
        "time": np.float64,
        "player1": np.int8,
        "player2": np.int8,
        "player1won": np.int8
    })
    
    
    
    match_data_path = os.path.join(os.path.dirname(__file__), "../../data/match_data.csv")
    sim_data_path = os.path.join(os.path.dirname(__file__), "../../data/sim_data.csv")
    
    
    match_data.to_csv(match_data_path, index=False)
    sim_data.to_csv(sim_data_path, index=False)


if __name__ == "__main__":

    from scripts.systems.ou_process import f,g
    
    def like(x,A):
        P = np.exp(A@x)
        Z = np.sum(P)
        P = P / Z
        return P
    
    # Example usage:
    num_of_players = 2
    num_of_matches = 100
    t0 = 0.0
    tf = 10.0
    dt = 0.001
    
    # Fix Randomness
    seed = 1
    
    # Initial condition
    mean_init = 0.0
    std_init = np.sqrt(2)
    
    # Generate random initial values for each simulation
    S0 = np.random.normal(mean_init, std_init, num_of_players)
    S0 = np.zeros(num_of_players)
    
    p_sde = (2.0, 0.0, np.sqrt(2))
    p_sde_sim = (t0, tf, dt, S0, f, g, p_sde)
    
    # A matrix for win-loss-draw likelihood
    A = np.array([[1, -1], [0.0, -0.0], [-1, 1]])
    
    p_like = (A,) 
    
    possible_outcomes = [1,0,-1]
    
    """
    Generate Data
    """
    
    gen_data(num_of_players,num_of_matches,possible_outcomes,seed,p_sde_sim,like,p_like)