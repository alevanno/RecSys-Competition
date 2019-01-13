import numpy as np
from pathlib import Path
import pandas as pd
from scipy.sparse import coo_matrix

train_path = Path("/home/andrea/Scrivania/RS competition/RecSys-Competition/data")/"train.csv"
target_path = Path("/home/andrea/Scrivania/RS competition/RecSys-Competition/data")/"target_playlists.csv"
tracks_path = Path("/home/andrea/Scrivania/RS competition/RecSys-Competition/data")/"tracks.csv"

class Data_matrix_utility(object):

    def build_icm_matrix(self):   #for now it works only for URM
        data = pd.read_csv(tracks_path)
        n_tracks = data.nunique().get('track_id')
        max_album_id = data.sort_values(by=['album_id'], ascending=False)['album_id'].iloc[0]
        max_artist_id = data.sort_values(by=['artist_id'], ascending=False)['artist_id'].iloc[0]

        track_array = self.extract_array_from_dataFrame(data, ['album_id',\
                                'artist_id','duration_sec'])
        album_array = self.extract_array_from_dataFrame(data, ['track_id',\
                                'artist_id','duration_sec' ])
        artist_array = self.extract_array_from_dataFrame(data, ['track_id',\
                                    'album_id','duration_sec'])
        artist_array = artist_array + max_album_id + 1
        attribute_array = np.concatenate((album_array, artist_array))
        extended_track_array = np.concatenate((track_array,track_array))
        implicit_rating = np.ones_like(np.arange(len(extended_track_array)))
        n_attributes = max_album_id + max_artist_id + 2
        icm = coo_matrix((implicit_rating, (extended_track_array, attribute_array)), \
                            shape=(n_tracks, n_attributes))
        return icm

    def build_urm_matrix(self):
        data = pd.read_csv(train_path)

        n_playlists = data.nunique().get('playlist_id')
        n_tracks = data.nunique().get('track_id')

        playlists_array = self.extract_array_from_dataFrame(data, ['track_id'])
        track_array = self.extract_array_from_dataFrame(data, ['playlist_id'])
        implicit_rating = np.ones_like(np.arange(len(track_array)))
        urm = coo_matrix((implicit_rating, (playlists_array, track_array)), \
                            shape=(n_playlists, n_tracks))
        return urm

    def get_target_list(self):
        targets_df = pd.read_csv(target_path)
        return targets_df.get_values().squeeze()

    def extract_array_from_dataFrame(self, data, columns_list_to_drop):
        array = data.drop(columns=columns_list_to_drop).get_values()
        return array.T.squeeze() #transform a nested array in array and transpose it

    def user_to_neglect(self, intensity_interaction='few', n_interactions=30):
        data = pd.read_csv(train_path)
        playlists_array = set(self.extract_array_from_dataFrame(data, ['track_id']))
        playlists_df = data.drop(columns=['track_id'])
        playlist_count = playlists_df['playlist_id'].value_counts()

        user_to_neglect = []
        for user in playlists_array:
            if intensity_interaction == 'few':
                if playlist_count[user] < n_interactions:
                    user_to_neglect.append(user)
            else:
                if playlist_count[user] >= n_interactions:
                    user_to_neglect.append(user)
        print("Neglected " + str(len(user_to_neglect)) + " users")
        return user_to_neglect

    def get_train_dataframe(self):
        return pd.read_csv(train_path)

    def get_target_dataframe(self):
        return pd.read_csv(target_path)

    def get_tracks_dataframe(self):
        return pd.read_csv(tracks_path)
    """
    def item_to_neglect(self):
        data = pd.read_csv(train_path)
        
        tracks_df = data.drop(columns=['playlist_id'])
        track_count = tracks_df['track_id'].value_counts()
        n_tracks = data.nunique().get('track_id')
        item_to_neglect = []
        for i in range(n_tracks):
            if track_count[i] < 30:
                item_to_neglect.append(i)
        print(len(item_to_neglect))
        return item_to_neglect
    """
