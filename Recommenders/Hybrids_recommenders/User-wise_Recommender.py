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
from Recommenders.CBF_and_CF_Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender

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
        self.elastic_new = MultiThreadSLIM_ElasticNet(self.URM_train)
        self.elastic_new.fit(alpha=0.0008868749995645901, l1_penalty=1.8986406043137196e-06,
                        l2_penalty=0.011673969837199876, topK=200)

        self.cbf_new = ItemKNNCBFRecommender(ICM=self.ICM, URM_train=self.URM_train)
        self.cbf_new.fit(topK=50, shrink=100, feature_weighting="TF-IDF")

        self.item_based = Item_based_CF_recommender(self.URM_train)
        self.item_based.fit(topK=150, shrink=20)

        self.user_based = User_based_CF_recommender(self.URM_train)
        self.user_based.fit(topK=180, shrink=2)

        self.graph = RP3betaRecommender(self.URM_train)
        self.graph.fit(topK=100, alpha=0.95, beta=0.3)

        self.bpr = SLIM_BPR_Cython(self.URM_train)
        self.bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

        self.elastic_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=self.URM_train, Similarity_1=self.elastic_new.W_sparse,
                                                            Similarity_2=self.cbf_new.W_sparse)
        self.elastic_hybrid.fit(alpha=0.95, beta=0.05, topK=250)

        self.item_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=self.URM_train, Similarity_1=self.item_based.W_sparse,
                                                         Similarity_2=self.cbf_new.W_sparse)
        self.item_hybrid.fit(alpha=0.8, beta=0.2, topK=150)

        self.graph_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=self.URM_train, Similarity_1=self.graph.W_sparse,
                                                          Similarity_2=self.cbf_new.W_sparse)
        self.graph_hybrid.fit(alpha=0.97, beta=0.03, topK=200)

        self.bpr_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=self.URM_train, Similarity_1=self.bpr.W_sparse,
                                                        Similarity_2=self.cbf_new.W_sparse)
        self.bpr_hybrid.fit(alpha=0.55, beta=0.45, topK=300)

        self.light_FM = LightFM_recommender(self.URM_train, self.ICM)
        self.light_FM.fit(no_components=50, item_alpha=1e-05, user_alpha=0.0001, epochs=90)

        plain_dict = {}
        plain_dict[elastic_new] = 50.0
        plain_dict[bpr] = 4.5
        plain_dict[cbf_new] = 6.5
        plain_dict[graph] = 10.0
        self.best_rec = Linear_combination_scores_recommender(urm_csr=self.URM_train, rec_dictionary=plain_dict)

        second_dict = {}
        second_dict[elastic_new] = 85.69524017409397
        second_dict[bpr] = 8.809832021747436
        second_dict[graph] = 10.963047273471327
        second_dict[cbf_new] = 5.9584783079642
        second_dict[item_based] = 6.590684520042074
        second_dict[user_based] = 0.1824636850615924
        second_dict[light_FM] = 6.3985793332282235
        self.second_rec = Linear_combination_scores_recommender(self.URM_train, rec_dictionary=second_dict)

        third_dict = {}
        third_dict[elastic_new] = 79.42214197079755
        third_dict[bpr] = 9.940183762075806
        third_dict[graph] = 9.616821564600901
        third_dict[item_based] = 9.88666288484462
        self.third_rec = Linear_combination_scores_recommender(self.URM_train, rec_dictionary=third_dict)

        fourth_dict = {}
        fourth_dict[elastic_new] = 77.77332812580758
        fourth_dict[bpr] = 9.575116156481108
        fourth_dict[graph] = 9.944083729532741
        fourth_dict[item_based] = 0.9830426465769995
        self.fourth_rec = Linear_combination_scores_recommender(self.URM_train, rec_dictionary=fourth_dict)

        fifth_dict = {}
        fifth_dict[elastic_new] = 77.23506889106422
        fifth_dict[bpr] = 9.929831307151332
        fifth_dict[graph] = 0.3671828951505075
        fifth_dict[item_based] = 9.376820981762133
        fifth_dict[user_based] = 1.0856382594992606
        self.fifth_rec = Linear_combination_scores_recommender(self.URM_train, rec_dictionary=fifth_dict)



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
    block_size = int(len(profile_length) * 0.10)
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

    elastic_new = MultiThreadSLIM_ElasticNet(urm_train)
    elastic_new.fit(alpha=0.0008868749995645901, l1_penalty=1.8986406043137196e-06,
                    l2_penalty=0.011673969837199876, topK=200)

    cbf_new = ItemKNNCBFRecommender(ICM=icm_complete, URM_train=urm_train)
    cbf_new.fit(topK=50, shrink=100, feature_weighting="TF-IDF")

    item_based = Item_based_CF_recommender(urm_train)
    item_based.fit(topK=150, shrink=20)

    user_based = User_based_CF_recommender(urm_train)
    user_based.fit(topK=180, shrink=2)

    graph = RP3betaRecommender(urm_train)
    graph.fit(topK=100, alpha=0.95, beta=0.3)

    bpr = SLIM_BPR_Cython(urm_train)
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

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

    light_FM = LightFM_recommender(urm_train, icm_complete)
    light_FM.fit(no_components=50, item_alpha=1e-05, user_alpha=0.0001, epochs=90)

    plain_dict = {}
    plain_dict[elastic_new] = 50.0
    plain_dict[bpr] = 4.5
    plain_dict[cbf_new] = 6.5
    plain_dict[graph] = 10.0
    best_rec = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=plain_dict)
    print(best_rec.evaluateRecommendations(URM_test=urm_test, at=10))

    second_dict = {}
    second_dict[elastic_new] = 85.69524017409397
    second_dict[bpr] = 8.809832021747436
    second_dict[graph] = 10.963047273471327
    second_dict[cbf_new] = 5.9584783079642
    second_dict[item_based] = 6.590684520042074
    second_dict[user_based] = 0.1824636850615924
    second_dict[light_FM] = 6.3985793332282235
    second_rec = Linear_combination_scores_recommender(urm_train, rec_dictionary=second_dict)
    print("Recommender2")
    print(second_rec.evaluateRecommendations(URM_test=urm_test, at=10))

    third_dict = {}
    third_dict[elastic_new] = 79.42214197079755
    third_dict[bpr] = 9.940183762075806
    third_dict[graph] = 9.616821564600901
    third_dict[item_based] = 9.88666288484462
    third_rec = Linear_combination_scores_recommender(urm_train, rec_dictionary=third_dict)
    print("Recommender3")
    print(third_rec.evaluateRecommendations(URM_test=urm_test, at=10))

    fourth_dict = {}
    fourth_dict[elastic_new] = 77.77332812580758
    fourth_dict[bpr] = 9.575116156481108
    fourth_dict[graph] = 9.944083729532741
    fourth_dict[item_based] = 0.9830426465769995
    fourth_rec = Linear_combination_scores_recommender(urm_train, rec_dictionary=fourth_dict)
    print("Recommender4")
    print(fourth_rec.evaluateRecommendations(URM_test=urm_test, at=10))

    fifth_dict = {}
    fifth_dict[elastic_new] = 77.23506889106422
    fifth_dict[bpr] = 9.929831307151332
    fifth_dict[graph] = 0.3671828951505075
    fifth_dict[item_based] = 9.376820981762133
    fifth_dict[user_based] = 1.0856382594992606
    fifth_rec = Linear_combination_scores_recommender(urm_train, rec_dictionary=fifth_dict)
    print("Recommender5")
    print(fifth_rec.evaluateRecommendations(URM_test=urm_test, at=10))




    profile_length = np.ediff1d(urm_train.indptr)
    block_size = int(len(profile_length) * 0.10)
    sorted_users = np.argsort(profile_length)


    MAP_best_per_group = []
    MAP_second_per_group = []
    MAP_third_per_group = []
    MAP_fourth_per_group = []
    MAP_fifth_per_group = []
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

        results, _ = evaluator_test.evaluateRecommender(best_rec)
        MAP_best_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(second_rec)
        MAP_second_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(third_rec)
        MAP_third_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(fourth_rec)
        MAP_fourth_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(fifth_rec)
        MAP_fifth_per_group.append(results[cutoff]["MAP"])


    pyplot.plot(MAP_best_per_group, label="best")
    pyplot.plot(MAP_second_per_group, label="second")
    pyplot.plot(MAP_third_per_group, label="third")
    pyplot.plot(MAP_fourth_per_group, label="fourth")
    pyplot.plot(MAP_fifth_per_group, label="fifth")
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()


