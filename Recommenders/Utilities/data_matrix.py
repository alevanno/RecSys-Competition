import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix

train_path = Path("data")/"train.csv"
target_path = Path('data')/'target_playlists.csv'

################################################################################
class Data_matrix_utility(object):
    def __init__(self, path):
        self.train_path = path

    def build_matrix(self):   #for now it works only for URM
        data = pd.read_csv(self.train_path)
        n_playlists = data.nunique().get('playlist_id')
        n_tracks = data.nunique().get('track_id')

        playlists_array = self.extract_array_from_dataFrame(data, ['track_id'])
        track_array = self.extract_array_from_dataFrame(data, ['playlist_id'])
        implicit_rating = np.ones_like(np.arange(len(track_array)))
        urm = coo_matrix((implicit_rating, (playlists_array, track_array)), \
                            shape=(n_playlists, n_tracks))
        return urm

    def extract_array_from_dataFrame(self, data, columns_list_to_drop):
        array = data.drop(columns=columns_list_to_drop).get_values()
        return array.T.squeeze() #transform a nested array in array and transpose it
################################################################################