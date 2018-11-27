import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix
from Compute_Similarity_Python import Compute_Similarity_Python
from evaluation_function import evaluate_algorithm
from data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot

tracks_path = Path("data")/"tracks.csv"
target_path = Path('data')/'target_playlists.csv'
train_path = Path("data")/"train.csv"

"""
Here a CBF recommender is implemented
"""
################################################################################
class Data_matrix_utility(object):
    def __init__(self, tracks_path, train_path):
        self.tracks_path = tracks_path
        self.train_path = train_path

    def build_icm_matrix(self):   #for now it works only for URM
        data = pd.read_csv(self.tracks_path)
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

################################################################################
class CBF_recommender(object):
    def __init__(self, urm_csr, icm_csr):
        self.icm_csr = icm_csr
        self.urm_csr = urm_csr

    def fit(self, topK=50, shrink=100, normalize = True, similarity = "cosine"):
        similarity_object = Compute_Similarity_Python(self.icm_csr.transpose(), shrink=shrink,\
                                                  topK=topK, normalize=normalize,\
                                                  similarity = similarity)
#I pass the transpose of urm to calculate the similarity between tracks.
#I obtain a similarity matrix of dimension = number of tracks * number_of_tracks
        self.similarity_csr = similarity_object.compute_similarity()

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        target_profile = self.urm_csr.getrow(target_id)
        scores = self.similarity_csr.dot(target_profile.transpose()).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(target_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:n_tracks]

    def filter_seen(self, target_id, scores):
        start_pos = self.urm_csr.indptr[target_id] #extracts the column in which the target start
        end_pos = self.urm_csr.indptr[target_id+1] #extracts the column in which the target ends
        target_profile = self.urm_csr.indices[start_pos:end_pos] #extracts the columns indexes
                                #in which we can find the non zero values in target
        scores[target_profile] = -np.inf
        return scores
################################################################################

def provide_recommendations(urm, icm):
    recommendations = {}
    urm_csr = urm.tocsr()
    icm_csr = icm.tocsr()
    targets_df = pd.read_csv(target_path)
    targets_array = targets_df.get_values().squeeze()
    recommender = CBF_recommender(urm_csr, icm_csr)
    recommender.fit(shrink=2, topK=180)
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target,n_tracks=10)

    with open('tuned_CBF_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility(tracks_path, train_path)
    icm_complete = utility.build_icm_matrix()
    urm_complete = utility.build_urm_matrix()
    #provide_recommendations(urm_complete, icm_complete)

    urm_train, urm_test = train_test_holdout(URM_all = urm_complete)
    recommender = CBF_recommender(urm_train, icm_complete)

    K_values = [100]
    K_results = []
    for k in K_values:
        recommender.fit(topK=k,shrink=3)
        evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
        print("k= " + str(k))
        print(evaluation_metrics)
        #K_results.append(evaluation_metrics["MAP"])
    """
    shrink_value = [x for x in range(10)]
    shrink_result = []
    for value in shrink_value:
        print('Evaluating shrink = ' + str(value))
        recommender.fit(topK = 100,shrink=value)
        evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
        print(evaluation_metrics)
        #shrink_results.append(evaluation_metrics["MAP"])

    #pyplot.plot(K_values, K_results)
    pyplot.plot(shrink_values, shrink_results)
    pyplot.ylabel('MAP')
    pyplot.xlabel('TopK')
    pyplot.show()
    """
