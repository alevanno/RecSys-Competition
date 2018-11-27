import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix

train_path = Path('data')/'train.csv'
target_path = Path('data')/'target_playlists.csv'
NUMBER_RECOMMENDATION = 10

def build_matrix():
    data = pd.read_csv(train_path)
    n_playlists = data.nunique().get('playlist_id')
    n_tracks = data.nunique().get('track_id')

    playlists_array = extract_array_from_dataFrame(data, ['track_id'])
    track_array = extract_array_from_dataFrame(data, ['playlist_id'])
    implicit_rating = np.ones_like(np.arange(len(track_array)))
    urm = coo_matrix((implicit_rating, (playlists_array, track_array)), shape=(n_playlists, n_tracks))
    return urm

def extract_array_from_dataFrame(data, columns_list_to_drop):
    array = data.drop(columns=columns_list_to_drop).get_values()
    return array.T.squeeze() #transform a nested array in array and transpose it

def recommendation(urm_csr, target_playlist):
    recommendation = set()
    target_vector = urm_csr.getrow(target_playlist)
    seen_item = target_vector.nonzero()[1]
    dot_vector = urm_csr.dot(target_vector.T)
    dot_vector[target_playlist, 0] = -1 #I cannot select the target_playlist
    dot_vector = dot_vector.toarray().ravel()
    dot_vector = np.argsort(dot_vector)[::-1]

    index = 0
    while len(recommendation) < NUMBER_RECOMMENDATION:
        suggested_songs = songs_in_a_playlist(urm_csr, dot_vector[index], seen_item)
        index += 1
        for song in suggested_songs:
            if len(recommendation) < NUMBER_RECOMMENDATION:
                recommendation.add(song)
    return recommendation

def songs_in_a_playlist(urm_csr, playlist_id, seen_item):
    suggestion = list()
    playlist_row = urm_csr.getrow(playlist_id)
    non_zero_indexes = playlist_row.nonzero()[1]
#    for song in non_zero_indexes:
#        if song in seen_item:
#            suggestion.append(song)
    return non_zero_indexes[:10]

def CF_recommender(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_df = pd.read_csv(target_path)
    targets_array = targets_df.get_values().squeeze()

    for target in targets_array:
        recommendations[target] = recommendation(urm_csr, target)

    with open('CFnaive.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    CF_recommender(build_matrix())

"""
Results:
- MAP on Public Test Set = 0.01540
"""
