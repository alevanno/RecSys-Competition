from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Base.Evaluation.Evaluator import SequentialEvaluator
from Recommenders.ML_recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.CBF_and_CF_Recommenders.Item_based_CF_Recommender import Item_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender
from Recommenders.Hybrids_recommenders.Linear_combination_scores_recommender import Linear_combination_scores_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.CBF_and_CF_Recommenders.Content_based_filtering import CBF_recommender

from Recommenders.CBF_and_CF_Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.Hybrids_recommenders.LightFM_recommender import LightFM_recommender
from Recommenders.ML_recommenders.FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg

import numpy as np
import matplotlib.pyplot as pyplot

class User_wise_Recommender(object):
    def __init__(self, urm_csr, icm, group_dict):
        self.group_dict = group_dict
        self.URM_train = urm_csr
        self.ICM = icm

    def fit(self):
        elastic = MultiThreadSLIM_ElasticNet(self.URM_train)
        elastic.fit(alpha=0.0008868749995645901, l1_penalty=1.8986406043137196e-06,
                    l2_penalty=0.011673969837199876, topK=200)

        graph = RP3betaRecommender(self.URM_train)
        graph.fit(topK=100, alpha=0.95, beta=0.3)

        bpr = SLIM_BPR_Cython(self.URM_train)
        bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

        cbf = CBF_recommender(urm_csr=self.URM_train, icm_csr=icm_complete.tocsr())
        cbf.fit(topK=100, shrink=3)

        cbf_boosted = CFW_D_Similarity_Linalg(URM_train=self.URM_train, ICM=icm_complete, S_matrix_target=elastic.W_sparse)
        cbf_boosted.fit(damp_coeff=0.1, add_zeros_quota=1.0, topK=50)

        plain_dict = {}
        plain_dict[elastic] = 50.0
        plain_dict[bpr] = 4.5
        plain_dict[cbf] = 6.5
        plain_dict[graph] = 10.0
        plain_dict[cbf_boosted] = 5.0
        self.content_rec = Linear_combination_scores_recommender(urm_csr=self.URM_train, rec_dictionary=plain_dict)

        other_dict = {}



    def recommend(self, target_id, n_tracks=None):
        user_group = self.get_user_group(target_id)
        if user_group <= 4:
            return self.content_rec.recommend(user_id_array=target_id, cutoff=n_tracks)
        else:
            return self.collaborative_rec.recommend(user_id_array=target_id, cutoff=n_tracks)

    def get_user_group(self, user_id):
        for group_id in range(len(self.group_dict)):
            if user_id in self.group_dict[group_id]:
                return group_id
        return len(self.group_dict) - 1

######################################################################################################################

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_array = utility.get_target_list()
    recommender = User_wise_Recommender(urm_csr=urm_csr, group_dict=group_formation(urm_csr), icm=icm_complete)
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target, n_tracks=10)

    with open('User-wise_recommendations_content_collaborative.csv',
              'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

def group_formation(urm_train):
    group_dict = {}

    profile_length = np.ediff1d(urm_train.indptr)
    block_size = int(len(profile_length) * 0.12)
    sorted_users = np.argsort(profile_length)

    for group_id in range(0, 10):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))
        users_in_group = sorted_users[start_pos:end_pos]
        group_dict[group_id] = users_in_group

    return group_dict

if __name__ == '__main__':
    utility = Data_matrix_utility()
    icm_complete = utility.build_icm_matrix().tocsr()
    urm_complete = utility.build_urm_matrix().tocsr()
    #provide_recommendations(utility.build_urm_matrix().tocsr())


    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)

    cbf_elastic_dict = {}
    elastic = MultiThreadSLIM_ElasticNet(urm_train)
    l1_value = 1e-05
    l2_value = 0.002
    k = 150
    elastic.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)

    cbf = ItemKNNCBFRecommender(ICM=icm_complete, URM_train=urm_train)
    cbf.fit(topK=50, shrink=100, feature_weighting="TF-IDF")
    cbf_elastic_dict[elastic] = 20.0
    cbf_elastic_dict[cbf] = 1.0
    #cbf_elastic_rec = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=cbf_elastic_dict)

    cbf_bpr_dict = {}
    bpr = SLIM_BPR_Cython(urm_train)
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)
    cbf_bpr_dict[bpr] = 1.2
    cbf_bpr_dict[cbf] = 1.0
    #cbf_bpr_rec = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=cbf_bpr_dict)

    user_rec = User_based_CF_recommender(urm_train)
    user_rec.fit(shrink=2, topK=180)

    item_rec = Item_based_CF_recommender(urm_train)
    item_rec.fit(topK=150, shrink=20)

    graph_rec = RP3betaRecommender(urm_train)
    graph_rec.fit(topK=100, alpha=0.95, beta=0.3)

    cbf_boosted = CFW_D_Similarity_Linalg(URM_train=urm_train, ICM=icm_complete, S_matrix_target=elastic.W_sparse)
    cbf_boosted.fit(damp_coeff=0.1, add_zeros_quota=1.0, topK=50)

    light_FM = LightFM_recommender(urm_train, icm_complete)
    light_FM.fit(no_components=50, item_alpha=1e-05, user_alpha=0.0001, epochs=90)

    plain_dict = {}
    plain_dict[elastic] = 50.0
    plain_dict[bpr] = 4.5
    plain_dict[cbf] = 6.5
    plain_dict[graph_rec] = 10.0
    best_rec = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=plain_dict)

    alternative_elastic = MultiThreadSLIM_ElasticNet(urm_train)
    alternative_elastic.fit(alpha=0.0008868749995645901, l1_penalty=1.8986406043137196e-06,
                        l2_penalty=0.011673969837199876, topK=200)
    linear_dict = {}
    linear_dict[alternative_elastic] = 50.0
    linear_dict[bpr] = 4.5
    linear_dict[cbf] = 6.5
    linear_dict[graph_rec] = 10.0
    best_rec_alternative = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=plain_dict)




    profile_length = np.ediff1d(urm_train.indptr)
    block_size = int(len(profile_length) * 0.10)
    sorted_users = np.argsort(profile_length)

    MAP_elastic_per_group = []
    MAP_cbf_per_group = []
    MAP_bpr_per_group = []
    MAP_user_per_group = []
    MAP_item_per_group = []
    MAP_graph_per_group = []
    MAP_cbf_boosted_per_group = []
    MAP_light_FM_per_group = []
    MAP_best_per_group = []
    MAP_best_alternative_per_group = []
    cutoff = 10

    for group_id in range(0, 10):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]

        evaluator_test = SequentialEvaluator(urm_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

        results, _ = evaluator_test.evaluateRecommender(elastic)
        MAP_elastic_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(cbf)
        MAP_cbf_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(bpr)
        MAP_bpr_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(user_rec)
        MAP_user_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(item_rec)
        MAP_item_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(graph_rec)
        MAP_graph_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(cbf_boosted)
        MAP_cbf_boosted_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(light_FM)
        MAP_light_FM_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(best_rec)
        MAP_best_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(best_rec_alternative)
        MAP_best_alternative_per_group.append(results[cutoff]["MAP"])

    pyplot.plot(MAP_elastic_per_group, label="elastic")
    pyplot.plot(MAP_cbf_per_group, label="cbf")
    pyplot.plot(MAP_bpr_per_group, label="bpr")
    pyplot.plot(MAP_user_per_group, label="user")
    pyplot.plot(MAP_item_per_group, label="item")
    pyplot.plot(MAP_graph_per_group, label="graph")
    pyplot.plot(MAP_cbf_boosted_per_group, label="cbf_boosted")
    pyplot.plot(MAP_light_FM_per_group, label="light_FM")
    pyplot.plot(MAP_best_per_group, label="best")
    pyplot.plot(MAP_best_alternative_per_group, label="best_alternative")
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()

