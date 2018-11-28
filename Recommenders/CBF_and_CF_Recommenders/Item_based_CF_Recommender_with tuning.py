import numpy as np
from Recommenders.Utilities.Compute_Similarity_Python import Compute_Similarity_Python
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Utilities.data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot
from Recommenders.Utilities.data_matrix import Data_matrix_utility

"""
Here an Item-based CF recommender is implemented
"""

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
    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all = urm_complete)
    recommender = Item_based_CF_recommender(urm_train)

    '''K_results = []
    for k in range(45,100,5): 
        print('Evaluating topK = ' + str(k))
        recommender.fit(topK=k,shrink=20)
        evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
        K_results.append(evaluation_metrics["MAP"])

    #best with topk=150+ and shrink = 0 (0.1032)
    #best with topk=85 and shrink = 20 (0.1075)
    '''
    shrink_result = []
    for value in range(15,21):
        print('Evaluating shrink = ' + str(value))
        recommender.fit(shrink=value)
        evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
        shrink_result.append(evaluation_metrics["MAP"])
        
    #best with 19-21 and default topk (0.1071)
    
    
    '''
    pyplot.plot(K_values, K_results)
    pyplot.ylabel('MAP')
    pyplot.xlabel('TopK')
    pyplot.show()
    '''
    
###############################################################################

