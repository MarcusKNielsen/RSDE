import pandas as pd
#from datetime import datetime
import numpy as np


class data_container:
    def __init__(self):
        self.dataframe = pd.DataFrame(columns=["match_id","time", "team1", "team2"])
        self.matches = {}

    def add_match(self, match_id, time, team1, team2):
        # Convert time to datetime if it's not already
        #time = pd.to_datetime(time)

        # Create new row
        new_match = pd.DataFrame([{"match_id": match_id,"time": time, "team1": team1, "team2": team2}])

        # Append new row to DataFrame
        self.dataframe = pd.concat([self.dataframe, new_match], ignore_index=True)
    
    def sort(self):
        self.dataframe = self.dataframe.sort_values(by="time", ascending=True)


class match:
    def __init__(self, match_id, datetime, team1, team2, likelihood_type, result):
        self.match_id = match_id
        self.datetime = datetime
        self.team1 = team1
        self.team2 = team2
        self.likelihood_type = likelihood_type
        self.result = result


#specific_time = pd.to_datetime("07-02-2025 16:00", format="%d-%m-%Y %H:%M")

data_test = data_container()
data_test.add_match(1,0.2, "A", "B")
data_test.add_match(2,0.3, "C", "B")
data_test.add_match(3,0.1,  "A", "B")

print(data_test.dataframe)

data_test.sort()

print(data_test.dataframe)

m = match(1,0.1,"A","B","score",(1,1))

"""
Load data
"""
#import pandas as pd

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

data = data_container()

data.dataframe[['time','team1','team2']] = raw_data[['time','player1','player2']]
data.dataframe['match_id'] = data.dataframe.index


#%%











