import pandas as pd
import numpy as np

"""
Build class which hold all player information based in data
"""

class PlayerData:
    def __init__(self, player_id, times, number_of_states):
        self.player_id = player_id
        self.times = np.array(times)
        self.data_matrix = np.zeros((len(times), number_of_states))
        self.matches_played = 0
        self.matches_in_total = len(times)

    def __repr__(self):
        return f"PlayerData(player_id={self.player_id}, matches={len(self.times)}, matrix_shape={self.data_matrix.shape})"

class PlayerInfo:
    def __init__(self, df, number_of_states):
        self.df = df
        self.number_of_states = number_of_states
        self.players = {}
        self.num_players = 0
        self._process_data()

    def _process_data(self):
        unique_players = np.unique(self.df[['player1', 'player2']].values)
        self.num_players = len(unique_players)
        for player in unique_players:
            times = self.df[(self.df['player1'] == player) | (self.df['player2'] == player)]['time'].values
            self.players[player] = PlayerData(player, times, self.number_of_states)

    def get_player_data(self, player_id):
        return self.players.get(player_id, None)

if __name__ == "__main__":

    """
    Load data
    """
    
    # Define file path
    path = "/home/max/Documents/DTU/MasterThesis/RSDE/data"
    filename = "match_data.csv"
    
    # Combine path and filename
    file_path = f"{path}/{filename}"
    
    # Read CSV into a DataFrame1
    data = pd.read_csv(file_path)
    
    player_info = PlayerInfo(data, number_of_states=32)
    player0 = player_info.get_player_data(0)








