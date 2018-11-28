import numpy as np
from pathlib import Path
import os
import pandas as pd
from Recommenders.Utilities.Compute_Similarity_Python import Compute_Similarity_Python
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Utilities.data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot
from Recommenders.Utilities.data_matrix import Data_matrix_utility


"""
Here a CBF recommender is implemented
"""


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
    targets_array = utility.get_target_list()
    recommender = CBF_recommender(urm_csr, icm_csr)
    recommender.fit(shrink=2, topK=180)
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target,n_tracks=10)

    with open('tuned_CBF_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
    icm_complete = utility.build_icm_matrix()
    urm_complete = utility.build_urm_matrix()
    provide_recommendations(urm_complete, icm_complete)
    """
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
