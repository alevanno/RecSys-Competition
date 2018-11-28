import numpy as np
from Recommenders.Utilities.Compute_Similarity_Python import Compute_Similarity_Python
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Utilities.data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot
from Recommenders.Utilities.data_matrix import Data_matrix_utility


"""
Here a User-based CF recommender is implemented
"""

################################################################################
class User_Based_CF_recommender(object):

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

if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_complete = utility.build_matrix()
    print("URM: " + str(urm_complete.shape))
    urm_train, urm_test = train_test_holdout(URM_all = urm_complete)
    print("Train: " + str(urm_train.shape))
    print("Test: " + str(urm_test.shape))
    recommender = User_Based_CF_recommender(urm_train)

    K_values = [170,175,180,185]
    K_results = []
    for k in K_values:
        recommender.fit(topK=k,shrink=0)
        evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
        K_results.append(evaluation_metrics["MAP"])
    """
    shrink_value = [for x in range(10)]
    shrink_result = []
    for value in shrink_value:
        print('Evaluating shrink = ' + str(value))
        recommender.fit(topK = ,shrink=value)
        evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
        shrink_results.append(evaluation_metrics["MAP"])

    #pyplot.plot(K_values, K_results)
    pyplot.plot(shrink_values, shrink_results)
    pyplot.ylabel('MAP')
    pyplot.xlabel('TopK')
    pyplot.show()


###############################################################################
Results:
- Execution time = about 2 minutes
- MAP on Public Test Set: 0.08343
"""
