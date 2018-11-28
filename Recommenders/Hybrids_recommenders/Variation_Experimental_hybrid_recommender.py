import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix
from Compute_Similarity_Python import Compute_Similarity_Python
from sklearn.preprocessing import StandardScaler

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

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        target_profile = self.similarity_csr.getrow(target_id)
        scores = target_profile.dot(self.urm_csr).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(target_id, scores)

        # rank items
        scores.sort()
        new_scores = scores[::-1]
        return new_scores[:n_tracks]

    def filter_seen(self, target_id, scores):
        start_pos = self.urm_csr.indptr[target_id] #extracts the column in which the target start
        end_pos = self.urm_csr.indptr[target_id+1] #extracts the column in which the target ends
        target_profile = self.urm_csr.indices[start_pos:end_pos] #extracts the columns indexes
                                #in which we can find the non zero values in target
        scores[target_profile] = -np.inf
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

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        target_profile = self.urm_csr.getrow(target_id)
        scores = target_profile.dot(self.similarity_csr).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(target_id, scores)

        # rank items
        scores.sort()
        new_scores = scores[::-1]
        return new_scores[:n_tracks]

    def filter_seen(self, target_id, scores):
        start_pos = self.urm_csr.indptr[target_id] #extracts the column in which the target start
        end_pos = self.urm_csr.indptr[target_id+1] #extracts the column in which the target ends
        target_profile = self.urm_csr.indices[start_pos:end_pos] #extracts the columns indexes
                                #in which we can find the non zero values in target
        scores[target_profile] = -np.inf
        return scores
################################################################################
################################################################################
class Hybrid_recommender(object):
    def __init__(self, urm_csr):
        self.urm_csr = urm_csr
        self.item_based_rec = Item_based_CF_recommender(self.urm_csr)
        self.user_based_rec = User_based_CF_recommender(self.urm_csr)

    def fit(self):
        self.item_based_rec.fit(topK=150, shrink=20)
        self.user_based_rec.fit(topK=180, shrink=2)

    def standardize(self, array):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel()

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        item_based_scores = self.item_based_rec.recommend(target_id, n_tracks=n_tracks)
        user_based_scores = self.user_based_rec.recommend(target_id, n_tracks=n_tracks)
        item_based_std_scores = self.standardize(item_based_scores)
        user_based_std_scores = self.standardize(user_based_scores)
        scores = np.add(item_based_std_scores, user_based_std_scores)
        print(scores)
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

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_df = pd.read_csv(target_path)
    targets_array = targets_df.get_values().squeeze()
    recommender = Hybrid_recommender(urm_csr)
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target,n_tracks=10)

    with open('variation_experimental_hybrid_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility(train_path)
    provide_recommendations(utility.build_matrix())
