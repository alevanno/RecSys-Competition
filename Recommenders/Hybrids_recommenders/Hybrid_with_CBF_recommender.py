import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix
from Recommenders.Utilities.Compute_Similarity_Python import Compute_Similarity_Python
from sklearn.preprocessing import StandardScaler
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Utilities.data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender

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
    targets_array = utility.get_target_list()
    recommender = Hybrid_recommender(urm_csr, icm_csr)
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target,n_tracks=10)

    with open('hybrid_with_CBF_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
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
