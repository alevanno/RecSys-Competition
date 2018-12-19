from Recommenders.Utilities.Compute_Similarity_Python import Compute_Similarity_Python
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Utilities.data_splitter import train_test_holdout
import numpy as np

"""
Here an Item-based CF recommender is implemented
"""

################################################################################
class Item_based_CF_recommender(SimilarityMatrixRecommender, Recommender):
    def __init__(self, URM_train):
        super(Item_based_CF_recommender, self).__init__()
        self.URM_train = URM_train

    def fit(self, topK=50, shrink=100, normalize = True, similarity = "cosine"):
        similarity_object = Compute_Similarity_Python(self.URM_train, shrink=shrink,\
                                                  topK=topK, normalize=normalize,\
                                                  similarity = similarity)
#I pass the transpose of urm to calculate the similarity between playlists.
#I obtain a similarity matrix of dimension = number of playlists * number_of_playlists
        self.W_sparse = similarity_object.compute_similarity()

################################################################################

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_array = utility.get_target_list()
    recommender = Item_based_CF_recommender(urm_csr)
    recommender.fit(topK=150, shrink=20)
    recommendations_array = recommender.recommend(user_id_array=targets_array, cutoff=10)
    for index in range(len(recommendations_array)):
        recommendations[targets_array[index]] = recommendations_array[index]

    with open('item_based_CF_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
    #provide_recommendations(utility.build_urm_matrix())
    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)


    item_based = Item_based_CF_recommender(urm_train)
    item_based.fit(topK=150, shrink=20)
    #print("best item based score train")
    #print(item_based.evaluateRecommendations(URM_test=urm_train, at=10, exclude_seen=False))
    """
    for id in range(urm_train.shape[0]):
        scores = item_based.compute_item_score(id)
        #print("Minimum: " + str(scores[np.nonzero(scores)].min()))
        #print("Maximum: " + str(scores.max()))
        #print("mean: " + str(scores[np.nonzero(scores)].mean()))

    #print("best item based score test")
    print(item_based.evaluateRecommendations(URM_test=urm_test, at=10))
    """
    item = Item_based_CF_recommender(urm_train)
    item.fit(topK=120, shrink=15)
    print("item based score train")
    print(item.evaluateRecommendations(URM_test=urm_train, at=10, exclude_seen=False))
    print("item based score test")
    print(item.evaluateRecommendations(URM_test=urm_test, at=10))
    """
    topk_list = [5, 10, 50, 70, 100, 120, 150, 170, 200, 250, 300]
    recommender = Item_based_CF_recommender(urm_train)
    for k in topk_list:
        print("k=" + str(k))
        recommender.fit(topK=k, shrink=15)
        print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
    """

###############################################################################
