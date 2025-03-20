from src.hermite import nodes,vander
from src.ivp_solver import fun_wave
from scipy.integrate import solve_ivp
#from datetime import datetime
import pandas as pd
import numpy as np
import importlib
#from scripts.systems.ou_process import a,D,dadx,dDdx

class Model:
    def __init__(self,raw_data,number_of_nodes=16):
        self.time_anchor = 0.0
        self.raw_data = raw_data
        self.teams = {}
        self.ito_process = {}
        self.number_of_nodes = number_of_nodes

    def create_teams_dict_from_raw_data(self):
        
        # find all teams in raw data set
        unique_teams = model.raw_data['teams'].explode().unique().tolist()
        matches_played = model.raw_data['teams'].explode().value_counts().to_dict()
        
        # create teams dict
        for i in range(len(unique_teams)):
            team_name = unique_teams[i]
            self.teams[team_name] = Team(team_name)
            self.teams[team_name].density = {}
            self.teams[team_name].density['time'] = np.zeros(matches_played[team_name])
            self.teams[team_name].density['state'] = np.zeros([matches_played[team_name],self.number_of_nodes+2])
    
    def set_initial_condition(self,initial_condition):
        for team in model.teams.keys():
            model.teams[team].state = initial_condition.copy()
    
    def run_filter(model):
        
        #dynamically imported module
        module = importlib.import_module(f"systems.{model.ito_process['type']}")
        attributes = ["a", "D"]
        objects = {attr: getattr(module, attr) for attr in attributes}
        
        # Initialize grid
        N = model.number_of_nodes
        z,w = nodes(N,Prob=True)
        
        # Matrices based on Hermite Functions
        V,Vz = vander(z,Prob=True)
        Vinv = np.linalg.inv(V)
        Mz = Vinv.T @ Vinv
        Mzd = np.diag(Mz) 
        Dz = Vz @ Vinv
        
        # setup parameters
        p = model.ito_process['parameters']
        p1 = (z, Dz, Mzd, objects['a'], objects['D'], p)
        
        # initial condition
        initial_condition = np.zeros(N+2)
        initial_condition[1] = np.sqrt(2)
        bhat = np.zeros(N)
        bhat[0] = 1
        initial_condition[2:] = V@bhat
        
        model.set_initial_condition(initial_condition)
        
        """
        Run filter / Loop through data
        """
        
        for match_data in model.raw_data.itertuples():
            print(match_data.Index)
            
            """
            Extract players data
            """
            team1 = match_data.teams[0]
            team2 = match_data.teams[1]
            
            match_num_team1 = model.teams[team1].matches_played
            match_num_team2 = model.teams[team2].matches_played
            
            t1 = model.teams[team1].time
            y1 = model.teams[team1].state
        
            t2 = model.teams[team2].time
            y2 = model.teams[team2].state
        
            """
            Perform time update
            """
            tf = match_data.time
            
            if match_num_team1 != 0:
                tspan=[t1, tf]      
                res1 = solve_ivp(fun_wave, tspan, y1, args=(p1))
                y1 = res1['y'][:,-1]
                
            
            if match_num_team2 != 0:
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
            b2 = y2[2:]
            w2 = b2*b2
            
            # Construct grids
            x1 = s1*z+m1
            x2 = s2*z+m2
            
            # Extract outcome of match
            outcome = match_data.results
            
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
            
            model.teams[team1].density['time'][match_num_team1] = tf
            model.teams[team2].density['time'][match_num_team2] = tf
            
            model.teams[team1].time = tf
            model.teams[team2].time = tf
            
            model.teams[team1].density['state'][match_num_team1] = y1
            model.teams[team2].density['state'][match_num_team2] = y2

            model.teams[team1].state = y1
            model.teams[team2].state = y2

            model.teams[team1].matches_played += 1
            model.teams[team2].matches_played += 1
        

class Team:
    def __init__(self,name):
        self.name = name
        self.time = 0.0
        self.matches_played = 0
        

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


if __name__ == "__main__":

    """
    Load data
    """

    # Define file path
    path = "/home/max/Documents/DTU/MasterThesis/RSDE/data"
    filename = "match_data.csv"

    # Combine path and filename
    file_path = f"{path}/{filename}"

    # Read CSV into a DataFrame
    raw_csv_file = pd.read_csv(file_path, dtype={
        "sim_idx": np.int32,
        "time": np.float64,
        "player1": np.int8,
        "player2": np.int8,
        "player1won": np.int8
    })
    
    raw_csv_file['player1'] = raw_csv_file['player1'].map({1: 'Astralis', 0: 'Vitality'})
    raw_csv_file['player2'] = raw_csv_file['player2'].map({1: 'Astralis', 0: 'Vitality'})

    raw_data = pd.DataFrame(columns=["time", "teams", "match_type", "results"])
    raw_data['time']  = raw_csv_file['time']
    raw_data['teams'] = raw_csv_file[['player1', 'player2']].apply(list, axis=1)
    raw_data['match_type'] = 'win-loss-draw'
    raw_data['results'] = raw_csv_file['player1won']
    
    """
    Setup Model
    """
    
    # Initialize model based on data
    model = Model(raw_data)
    
    # Setup underlying Ito process
    model.ito_process['type'] = 'ou_process'
    model.ito_process['parameters'] = np.array([2.0,0.0,np.sqrt(2)])
    
    model.create_teams_dict_from_raw_data()
    
    model.run_filter()

    import matplotlib.pyplot as plt
    
    team1 = 'Vitality'
    t = model.teams[team1].density['time']
    m1 = model.teams[team1].density['state'][:,0]
    
    team2 = 'Astralis'
    t = model.teams[team2].density['time']
    m2 = model.teams[team2].density['state'][:,0]
    
    
    def get_density_of_player(team,model,x_large):

        N = model.number_of_nodes
        M = len(x_large)    

        z,w = nodes(N,Prob=True)
        
        # Matrices based on Hermite Functions
        V,Vz = vander(z,Prob=True)
        Vinv = np.linalg.inv(V)

        t = model.teams[team].density['time']
        m = model.teams[team].density['state'][:,0]
        s = model.teams[team].density['state'][:,1]
        b = model.teams[team].density['state'][:,2:]
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

    for idx,team in enumerate(model.teams.keys()):
        t,u_large = get_density_of_player(team,model,x_large)
        T,X = np.meshgrid(t,x_large)
        plt.figure()
        plt.pcolormesh(T,X,u_large.T)
        plt.plot(sim_data.time,sim_data.iloc[:,idx+1],color="red")
        plt.xlabel("t: time")
        plt.ylabel("x: space")
        #plt.title(f"Player = {player}")
        plt.xlim([t[0],t[-1]])
        plt.tight_layout()
        plt.show()