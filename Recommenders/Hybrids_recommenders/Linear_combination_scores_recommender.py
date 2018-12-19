from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.CBF_and_CF_Recommenders.Item_based_CF_Recommender import Item_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.Content_based_filtering import CBF_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

import numpy as np
from sklearn.preprocessing import StandardScaler

class Linear_combination_scores_recommender(Recommender):
    def __init__(self, urm_csr, rec_dictionary, scale=False):
        super(Linear_combination_scores_recommender, self).__init__()
        self.URM_train = urm_csr
        self.rec_dict = rec_dictionary
        self.scale = scale

    def compute_item_score(self, user_id):
        scores = np.zeros(shape=(len(user_id), self.URM_train.shape[1]))

        for tuple in self.rec_dict.items():
            if self.scale:
                scores += self.standardize(tuple[0].compute_item_score(user_id)) * tuple[1]
            else:
                scores += tuple[0].compute_item_score(user_id) * tuple[1]

        return scores

    def standardize(self, array):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel()

    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores

####################################################################################################################
def provide_recommendations():
    recommendations = {}
    targets_array = utility.get_target_list()

    for target in targets_array:
        recommendations[target] = recommender.recommend(user_id_array=target, cutoff=10)

    with open('Linear_combination.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_complete = utility.build_urm_matrix()
    icm_complete = utility.build_icm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    rec_dictionary = {}

    item_based = Item_based_CF_recommender(urm_train)
    item_based.fit(topK=150, shrink=20)

    user_based = User_based_CF_recommender(urm_train)
    user_based.fit(topK=180, shrink=2)

    cbf = CBF_recommender(urm_csr=urm_train, icm_csr=icm_complete)
    cbf.fit(topK=180, shrink=2)

    elasticNet = MultiThreadSLIM_ElasticNet(urm_train)
    l1_value = 1e-05
    l2_value = 0.002
    k = 150
    elasticNet.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)

    bpr = SLIM_BPR_Cython(urm_train)
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

    rec_dictionary[item_based] = 5.0
    rec_dictionary[user_based] = 1.0
    rec_dictionary[cbf] = 1.4
    rec_dictionary[elasticNet] = 50.0
    rec_dictionary[bpr] = 2.0

    recommender = Linear_combination_scores_recommender(urm_csr=urm_complete.tocsr(), rec_dictionary=rec_dictionary)
    provide_recommendations()

    #recommender = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=rec_dictionary)
    #print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
