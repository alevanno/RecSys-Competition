import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix
from Compute_Similarity_Python import Compute_Similarity_Python
from sklearn.preprocessing import StandardScaler
from evaluation_function import evaluate_algorithm
from data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot

train_path = Path("data")/"train.csv"
tracks_path = Path("data")/"tracks.csv"
target_path = Path('data')/'target_playlists.csv'

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
class User_based_CF_recommender(object):
    def __init__(self, urm_csr):
        self.urm_csr = urm_csr

    def fit(self, topK=50, shrink=100, normalize = True, similarity = "cosine"):
        similarity_object = Compute_Similarity_Python(self.urm_csr.transpose(), shrink=shrink,\
                                                  topK=topK, normalize=normalize,\
                                                  similarity = similarity)
#I pass the transpose of urm to calculate the similarity between playlists.
#I obtain a similarity matrix of dimension = number of playlists * number_of_playlists
        self.similarity_csr = similarity_object.compute_similarity()

    def recommend(self, target_id):
        target_profile = self.similarity_csr.getrow(target_id)
        scores = target_profile.dot(self.urm_csr).toarray().ravel()
        return scores
################################################################################
################################################################################
class Item_based_CF_recommender(object):
    def __init__(self, urm_csr):
        self.urm_csr = urm_csr

    def fit(self, topK=50, shrink=100, normalize = True, similarity = "cosine"):
        similarity_object = Compute_Similarity_Python(self.urm_csr, shrink=shrink,\
                                                  topK=topK, normalize=normalize,\
                                                  similarity = similarity)
#I pass the transpose of urm to calculate the similarity between playlists.
#I obtain a similarity matrix of dimension = number of playlists * number_of_playlists
        self.similarity_csr = similarity_object.compute_similarity()

    def recommend(self, target_id):
        target_profile = self.urm_csr.getrow(target_id)
        scores = target_profile.dot(self.similarity_csr).toarray().ravel()
        return scores
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

    def recommend(self, target_id):
        target_profile = self.urm_csr.getrow(target_id)
        scores = self.similarity_csr.dot(target_profile.transpose()).toarray().ravel()
        return scores
################################################################################
################################################################################
class Hybrid_recommender(object):
    def __init__(self, urm_csr, icm_csr):
        self.urm_csr = urm_csr
        self.icm_csr = icm_csr
        self.item_based_rec = Item_based_CF_recommender(self.urm_csr)
        self.user_based_rec = User_based_CF_recommender(self.urm_csr)
        self.cbf_rec = CBF_recommender(self.urm_csr, self.icm_csr)

    def fit(self):
        self.item_based_rec.fit(topK=150, shrink=20)
        self.user_based_rec.fit(topK=180, shrink=2)
        self.cbf_rec.fit(topK=120, shrink=0)

    def standardize(self, array):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel()

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        item_based_scores = self.item_based_rec.recommend(target_id)
        user_based_scores = self.user_based_rec.recommend(target_id)
        cbf_scores = self.cbf_rec.recommend(target_id)
        item_based_std_scores = self.standardize(item_based_scores)
        user_based_std_scores = self.standardize(user_based_scores)
        cbf_std_scores = self.standardize(cbf_scores)
        scores = np.add(item_based_std_scores, user_based_std_scores, 2*cbf_std_scores)
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
##############################################################################

def provide_recommendations(urm, icm):
    recommendations = {}
    urm_csr = urm.tocsr()
    icm_csr = icm.tocsr()
    targets_df = pd.read_csv(target_path)
    targets_array = targets_df.get_values().squeeze()
    recommender = Hybrid_recommender(urm_csr, icm_csr)
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target,n_tracks=10)

    with open('hybrid_with_CBF_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility(tracks_path, train_path)
#    icm = utility.build_icm_matrix()
#    urm = utility.build_urm_matrix()
#    provide_recommendations(urm, icm)

    icm = utility.build_icm_matrix()
    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all = urm_complete)
    recommender = Hybrid_recommender(urm_train, icm.tocsr())

    recommender.fit()
    evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
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

"""
Result on public test set: 0.08637
"""
