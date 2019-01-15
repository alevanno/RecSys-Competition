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

    with open('Linear_combination_to_submit.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

def user_to_neglect(type = "under", group_id = 4):
    profile_length = np.ediff1d(urm_train.indptr)
    block_size = int(len(profile_length) * 0.10)
    sorted_users = np.argsort(profile_length)

    if type == "under":
        start_pos = 0
        end_pos = min((group_id + 1) * block_size, len(profile_length))
        user_to_neglect = sorted_users[start_pos:end_pos]
    else:
        start_pos = min((group_id + 1) * block_size, len(profile_length))
        end_pos = len(profile_length)
        user_to_neglect = sorted_users[start_pos:end_pos]

    return user_to_neglect

if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_complete = utility.build_urm_matrix().tocsr()
    icm_complete = utility.build_icm_matrix().tocsr()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)

    elastic_new = MultiThreadSLIM_ElasticNet(urm_train)
    elastic_new.fit(alpha=0.0008868749995645901, l1_penalty=1.8986406043137196e-06,
                    l2_penalty=0.011673969837199876, topK=200)


    graph = RP3betaRecommender(urm_train)
    graph.fit(topK=100, alpha=0.95, beta=0.3)

    bpr = SLIM_BPR_Cython(urm_train)
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

    cbf_new = ItemKNNCBFRecommender(ICM=icm_complete, URM_train=urm_train)
    cbf_new.fit(topK=50, shrink=100, feature_weighting="TF-IDF")

    plain_dict = {}
    plain_dict[elastic_new] = 50.0
    plain_dict[bpr] = 4.5
    plain_dict[cbf_new] = 6.5
    plain_dict[graph] = 10.0

    best_rec = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=plain_dict)


    print("best so far")
    print(best_rec.evaluateRecommendations(URM_test=urm_test, at=10))

    item_based = Item_based_CF_recommender(urm_train)
    item_based.fit(topK=150, shrink=20)

    user_based = User_based_CF_recommender(urm_train)
    user_based.fit(shrink=2, topK=180)

    elastic_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_train, Similarity_1=elastic_new.W_sparse,
                                                        Similarity_2=cbf_new.W_sparse)
    elastic_hybrid.fit(alpha=0.95, beta=0.05, topK=250)

    item_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_train, Similarity_1=item_based.W_sparse,
                                                     Similarity_2=cbf_new.W_sparse)
    item_hybrid.fit(alpha=0.8, beta=0.2, topK=150)

    graph_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_train, Similarity_1=graph.W_sparse,
                                                      Similarity_2=cbf_new.W_sparse)
    graph_hybrid.fit(alpha=0.97, beta=0.03, topK=200)

    bpr_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_train, Similarity_1=bpr.W_sparse,
                                                    Similarity_2=cbf_new.W_sparse)
    bpr_hybrid.fit(alpha=0.55, beta=0.45, topK=300)

    other_dict = {}

    elastic_list = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0,
                    10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0]
    item_list = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    bpr_list = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    graph_list = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                  6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    cbf_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 7.0, 7.0]

    for e, i, b, g, c in zip(elastic_list, item_list, bpr_list, graph_list, cbf_list):
        print("Config: elastic=" + str(e) + ", item=" + str(i) + ", BPR=" + str(b) + ", graph=" + str(g) +
              ", cbf=" + str(c))
        other_dict[elastic_hybrid] = e
        other_dict[item_hybrid] = i
        other_dict[bpr_hybrid] = b
        other_dict[graph_hybrid] = g
        other_dict[cbf_new] = c
        recommender = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=other_dict)
        print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
