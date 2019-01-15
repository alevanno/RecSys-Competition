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

import numpy as np
from sklearn.preprocessing import StandardScaler

class Linear_combination_scores_recommender(Recommender):
    def __init__(self, urm_csr, rec_dictionary, scale=False):
        super(Linear_combination_scores_recommender, self).__init__()
        self.URM_train = urm_csr
        self.rec_dict = rec_dictionary
        self.scale = scale

    def compute_item_score(self, user_id):
        self.filterTopPop_ItemsID = self.top_pop_item()
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
    #elastic_cbf_dictionary = {}
    #bpr_cbf_dictionary = {}

    #user_to_avoid = user_to_neglect(type="under", group_id=4)

    elastic = MultiThreadSLIM_ElasticNet(urm_train)
    l1_value = 1e-05
    l2_value = 0.002
    k = 150
    elastic.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)


    graph = RP3betaRecommender(urm_train)
    graph.fit(topK=100, alpha=0.95, beta=0.3)

    bpr = SLIM_BPR_Cython(urm_train)
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)


    cbf = CBF_recommender(urm_csr=urm_train, icm_csr=icm_complete.tocsr())
    cbf.fit(topK=100, shrink=3)

    light_FM = LightFM_recommender(urm_train, icm_complete)
    light_FM.fit(no_components=50, item_alpha=1e-05, user_alpha=0.0001, epochs=90, loss='bpr')


    plain_dict = {}
    plain_dict[elastic] = 50.0
    plain_dict[bpr] = 4.5
    plain_dict[cbf] = 6.5
    plain_dict[graph] = 10.0
    #plain_dict[light_FM] = 5.0

    best_rec = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=plain_dict)


    print("best so far")
    print(best_rec.evaluateRecommendations(URM_test=urm_test, at=10))
    print("Elimination of top pop: ")
    best_rec.filterTopPop = True
    print(best_rec.evaluateRecommendations(URM_test=urm_test, at=10))

    item_based = Item_based_CF_recommender(urm_train)
    item_based.fit(topK=150, shrink=20)

    user_based = User_based_CF_recommender(urm_train)
    user_based.fit(shrink=2, topK=180)

    #light_FM = LightFM_recommender(urm_train, icm_complete)
    #light_FM.fit(no_components=50, item_alpha=1e-05, user_alpha=0.0001, epochs=90, loss='bpr')

    try_dict = {}
    try_dict[elastic] = 85.69524017409397
    try_dict[bpr] = 8.809832021747436
    try_dict[graph] = 10.963047273471327
    try_dict[cbf] = 5.9584783079642
    try_dict[item_based] = 6.590684520042074
    try_dict[user_based] = 0.1824636850615924
    try_dict[light_FM] = 6.3985793332282235

    recommender1 = Linear_combination_scores_recommender(urm_train, rec_dictionary=try_dict)
    print("Recommender1")
    print(recommender1.evaluateRecommendations(URM_test=urm_test, at=10))

    cbf_new = ItemKNNCBFRecommender(ICM=icm_complete, URM_train=urm_train)
    cbf_new.fit(topK=50, shrink=100, feature_weighting="TF-IDF")

    elastic_new = MultiThreadSLIM_ElasticNet(urm_train)
    elastic_new.fit(alpha=0.0008868749995645901, l1_penalty=1.8986406043137196e-06,
                    l2_penalty=0.011673969837199876, topK=200)

    try_dict = {}
    try_dict[elastic_new] = 85.69524017409397
    try_dict[bpr] = 8.809832021747436
    try_dict[graph] = 10.963047273471327
    try_dict[cbf_new] = 5.9584783079642
    try_dict[item_based] = 6.590684520042074
    try_dict[user_based] = 0.1824636850615924
    try_dict[light_FM] = 6.3985793332282235

    recommender2 = Linear_combination_scores_recommender(urm_train, rec_dictionary=try_dict)
    print("Recommender2")
    print(recommender2.evaluateRecommendations(URM_test=urm_test, at=10))

#Recommender2 è il migliore per ora