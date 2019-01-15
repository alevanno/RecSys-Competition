from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.CBF_and_CF_Recommenders.Item_based_CF_Recommender import Item_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.Content_based_filtering import CBF_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ML_recommenders.FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender
from Recommenders.ML_recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.ML_recommenders.FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from Recommenders.Hybrids_recommenders.LightFM_recommender import LightFM_recommender
from Recommenders.CBF_and_CF_Recommenders.Item_based_CF_Recommender import Item_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.CBF_and_CF_Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender

import numpy as np
from sklearn.preprocessing import StandardScaler

class Linear_combination_scores_recommender(Recommender):
    def __init__(self, urm_csr, rec_dictionary, scale=False):
        super(Linear_combination_scores_recommender, self).__init__()
        self.URM_train = urm_csr
        self.rec_dict = rec_dictionary
        self.scale = scale

    def compute_item_score(self, user_id):
        #self.filterTopPop_ItemsID = self.top_pop_item()
        scores = np.zeros(shape=(len(user_id), self.URM_train.shape[1]))

        for tuple in self.rec_dict.items():
            if self.scale:
                shape = (len(user_id), self.URM_train.shape[1])
                scores += self.standardize(tuple[0].compute_item_score(user_id), shape) * tuple[1]
            else:
                scores += tuple[0].compute_item_score(user_id) * tuple[1]

        return scores

    def standardize(self, array, shape):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel().reshape(shape)

    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores

    def top_pop_item(self, cutoff=10):
        urm_t = self.URM_train.transpose()
        profile_length = np.ediff1d(urm_t.indptr)
        sorted_items = np.argsort(profile_length)[::-1]
        return sorted_items[0:cutoff]

####################################################################################################################
def provide_recommendations():
    recommendations = {}
    targets_array = utility.get_target_list()

    for target in targets_array:
        r = recommender.recommend(user_id_array=[target], cutoff=10)
        recommendations[target] = [item for sublist in r for item in sublist]

    with open('in_zona_cesarini.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))


if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_complete = utility.build_urm_matrix().tocsr()
    icm_complete = utility.build_icm_matrix().tocsr()

    elastic_new = MultiThreadSLIM_ElasticNet(urm_complete)
    elastic_new.fit(alpha=0.0008868749995645901, l1_penalty=1.8986406043137196e-06,
                    l2_penalty=0.011673969837199876, topK=200)

    graph = RP3betaRecommender(urm_complete)
    graph.fit(topK=100, alpha=0.95, beta=0.3)

    bpr = SLIM_BPR_Cython(urm_complete)
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

    cbf_new = ItemKNNCBFRecommender(ICM=icm_complete, URM_train=urm_complete)
    cbf_new.fit(topK=50, shrink=100, feature_weighting="TF-IDF")


    item_based = Item_based_CF_recommender(urm_complete)
    item_based.fit(topK=150, shrink=20)

    user_based = User_based_CF_recommender(urm_complete)
    user_based.fit(shrink=2, topK=180)

    elastic_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_complete, Similarity_1=elastic_new.W_sparse,
                                                        Similarity_2=cbf_new.W_sparse)
    elastic_hybrid.fit(alpha=0.95, beta=0.05, topK=250)

    item_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_complete, Similarity_1=item_based.W_sparse,
                                                     Similarity_2=cbf_new.W_sparse)
    item_hybrid.fit(alpha=0.8, beta=0.2, topK=150)

    graph_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_complete, Similarity_1=graph.W_sparse,
                                                      Similarity_2=cbf_new.W_sparse)
    graph_hybrid.fit(alpha=0.97, beta=0.03, topK=200)

    bpr_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_complete, Similarity_1=bpr.W_sparse,
                                                    Similarity_2=cbf_new.W_sparse)
    bpr_hybrid.fit(alpha=0.55, beta=0.45, topK=300)

    light_FM = LightFM_recommender(urm_complete, icm_complete)
    light_FM.fit(no_components=50, item_alpha=1e-05, user_alpha=0.0001, epochs=90)

    second_dict = {}
    second_dict[elastic_new] = 85.69524017409397
    second_dict[bpr] = 8.809832021747436
    second_dict[graph] = 10.963047273471327
    second_dict[cbf_new] = 5.9584783079642
    second_dict[item_based] = 6.590684520042074
    second_dict[user_based] = 0.1824636850615924
    second_dict[light_FM] = 6.3985793332282235
    recommender = Linear_combination_scores_recommender(urm_complete, rec_dictionary=second_dict)
    provide_recommendations()



