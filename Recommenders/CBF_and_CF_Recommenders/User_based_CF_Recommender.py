
import numpy as np
from Recommenders.Utilities.Compute_Similarity_Python import Compute_Similarity_Python
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.data_splitter import train_test_holdout

"""
Here a User-based CF recommender is implemented
"""

################################################################################
class User_based_CF_recommender(SimilarityMatrixRecommender, Recommender):
    def __init__(self, URM_train):
        super(User_based_CF_recommender, self).__init__()
        self.URM_train = URM_train

    def fit(self, topK=50, shrink=100, normalize = True, similarity = "cosine"):
        similarity_object = Compute_Similarity_Python(self.URM_train.transpose(), shrink=shrink,\
                                                  topK=topK, normalize=normalize,\
                                                  similarity = similarity)
#I pass the transpose of urm to calculate the similarity between playlists.
#I obtain a similarity matrix of dimension = number of playlists * number_of_playlists
        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, remove_top_pop_flag = False, remove_CustomItems_flag = False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False


        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self.compute_item_score(user_id_array)


        # if self.normalize:
        #     # normalization will keep the scores in the same range
        #     # of value of the ratings in dataset
        #     user_profile = self.URM_train[user_id]
        #
        #     rated = user_profile.copy()
        #     rated.data = np.ones_like(rated.data)
        #     if self.sparse_weights:
        #         den = rated.dot(self.W_sparse).toarray().ravel()
        #     else:
        #         den = rated.dot(self.W).ravel()
        #     den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
        #     scores /= den


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)


        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_CustomItems_flag:
            scores_batch = self._remove_CustomItems_on_scores(scores_batch)

        # scores_batch = np.arange(0,3260).reshape((1, -1))
        # scores_batch = np.repeat(scores_batch, 1000, axis = 0)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = ranking.tolist()


        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        return ranking_list

    def compute_item_score(self, user_id):
        return self.W_sparse[user_id].dot(self.URM_train).toarray()

################################################################################

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_array = utility.get_target_list()
    recommender = User_based_CF_recommender(urm_csr)
    recommender.fit(shrink=2, topK=180)
    recommendetions_array = recommender.recommend(user_id_array=targets_array, cutoff=10)
    for index in range(len(targets_array)):
        recommendations[targets_array[index]] = recommendetions_array[index]

    with open('user_based_CF_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
    #provide_recommendations(utility.build_urm_matrix())
    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    recommender = User_based_CF_recommender(urm_train)
    recommender.fit(topK=180, shrink=2)
    print("Best user based score: ")

    for id in range(urm_train.shape[0]):
        scores = recommender.compute_score_user_based(id)
        print("Minimum: " + str(scores[np.nonzero(scores)].min()))
        print("Maximum: " + str(scores.max()))
        print("mean: " + str(scores[np.nonzero(scores)].mean()))

    print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
    """
    topk_list = [50, 70, 100, 120, 150, 180, 200, 220, 250, 280, 300]
    user = User_based_CF_recommender(urm_train)
    for k in topk_list:
        print("k=" + str(k))
        user.fit(shrink=5, topK=k)
        print(user.evaluateRecommendations(URM_test=urm_test, at=10))
    """

###########################################################################################
