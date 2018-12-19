import numpy as np
from pathlib import Path
import os
import pandas as pd
from Recommenders.Utilities.Compute_Similarity_Python import Compute_Similarity_Python
from Recommenders.Utilities.data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender

"""
Here a CBF recommender is implemented
"""


################################################################################
class CBF_recommender(Recommender, SimilarityMatrixRecommender):
    def __init__(self, urm_csr, icm_csr):
        super(CBF_recommender, self).__init__()
        self.icm_csr = icm_csr
        self.URM_train= urm_csr

    def fit(self, topK=50, shrink=100, normalize = True, similarity = "cosine"):
        similarity_object = Compute_Similarity_Python(self.icm_csr.transpose(), shrink=shrink,\
                                                  topK=topK, normalize=normalize,\
                                                  similarity = similarity)
#I pass the transpose of urm to calculate the similarity between tracks.
#I obtain a similarity matrix of dimension = number of tracks * number_of_tracks
        self.W_sparse = similarity_object.compute_similarity()

################################################################################

def provide_recommendations(urm, icm):
    recommendations = {}
    urm_csr = urm.tocsr()
    icm_csr = icm.tocsr()
    targets_array = utility.get_target_list()
    recommender = CBF_recommender(urm_csr, icm_csr)
    recommender.fit(shrink=2, topK=180)
    for target in targets_array:
        recommendations[target] = recommender.recommend(user_id_array=target, cutoff=10)

    with open('tuned_CBF_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
    icm_complete = utility.build_icm_matrix()
    urm_complete = utility.build_urm_matrix()
    #provide_recommendations(urm_complete, icm_complete)

    urm_train, urm_test = train_test_holdout(URM_all = urm_complete)
    recommender = CBF_recommender(urm_train, icm_complete)

    K_values = [100]
    K_results = []
    for k in K_values:
        recommender.fit(topK=k,shrink=3)
        evaluation_metrics = recommender.evaluateRecommendations(URM_test=urm_test, at=10)
        print("k= " + str(k))
        print(evaluation_metrics)
        #K_results.append(evaluation_metrics["MAP"])
    """"
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