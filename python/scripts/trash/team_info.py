import pandas as pd
import numpy as np

"""
Build class which hold all team information based in data
"""

class teamData:
    def __init__(self, team_id, times, number_of_states):
        self.team_id = team_id
        self.times = np.array(times)
        self.data_matrix = np.zeros((len(times), number_of_states))
        self.matches_played = 0
        self.matches_in_total = len(times)

    def __repr__(self):
        return f"teamData(team_id={self.team_id}, matches={len(self.times)}, matrix_shape={self.data_matrix.shape})"

class teamInfo:
    def __init__(self, dataframe, number_of_states):
        self.dataframe = dataframe
        self.number_of_states = number_of_states
        self.teams_data = {}
        self.number_of_teams = 0
        self._process_data()

    def _process_data(self):
        # Extract unique team IDs from 'teams' column
        unique_teams = np.unique(np.concatenate(self.dataframe['teams'].values))
        
        self.number_of_teams = len(unique_teams)
        
        for team in unique_teams:
            # Get times where the team appears in 'teams'
            times = self.dataframe[self.dataframe['teams'].apply(lambda x: team in x)]['time'].values
            self.teams[team] = teamData(team, times, self.number_of_states)

    def get_team_data(self, team_id):
        return self.teams.get(team_id, None)

if __name__ == "__main__":

    """
    Load data
    """
    
    data = pd.DataFrame(columns=["time", "teams", "likelihood_type", "result"])

    # Define file path
    path = "/home/max/Documents/DTU/MasterThesis/RSDE/data"
    filename = "match_data.csv"

    # Combine path and filename
    file_path = f"{path}/{filename}"

    # Read CSV into a DataFrame
    raw_data = pd.read_csv(file_path, dtype={
        "sim_idx": np.int32,
        "time": np.float64,
        "player1": np.int8,
        "player2": np.int8,
        "player1won": np.int8
    })

    data['time']  = raw_data['time']
    data['teams'] = raw_data[['player1', 'player2']].apply(list, axis=1)
    data['likelihood_type'] = 'win-loss-draw'
    data['result'] = raw_data['player1won']
    
    team_info = teamInfo(data, number_of_states=32)
    team0 = team_info.get_team_data(0)








