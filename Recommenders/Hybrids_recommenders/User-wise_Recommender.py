from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Base.Evaluation.Evaluator import SequentialEvaluator
from Recommenders.ML_recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.CBF_and_CF_Recommenders.Item_based_CF_Recommender import Item_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender
from Recommenders.Hybrids_recommenders.Linear_combination_scores_recommender import Linear_combination_scores_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.CBF_and_CF_Recommenders.Content_based_filtering import CBF_recommender

import numpy as np
import matplotlib.pyplot as pyplot

class User_wise_Recommender(object):
    def __init__(self, urm_csr, icm, group_dict):
        self.group_dict = group_dict
        self.URM_train = urm_csr
        self.ICM = icm

    def fit(self):
        cbf_elastic_dict = {}
        elastic = MultiThreadSLIM_ElasticNet(self.URM_train)
        l1_value = 1e-05
        l2_value = 0.002
        k = 150
        elastic.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)
        cbf = CBF_recommender(urm_csr=self.URM_train, icm_csr=self.ICM)
        cbf.fit(topK=100, shrink=3)
        cbf_elastic_dict[elastic] = 20.0
        cbf_elastic_dict[cbf] = 1.0
        self.cbf_elastic_rec = Linear_combination_scores_recommender(urm_csr=self.URM_train, rec_dictionary=cbf_elastic_dict)

        cbf_bpr_dict = {}
        bpr = SLIM_BPR_Cython(self.URM_train)
        bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)
        cbf_bpr_dict[bpr] = 1.2
        cbf_bpr_dict[cbf] = 1.0
        self.cbf_bpr_rec = Linear_combination_scores_recommender(urm_csr=self.URM_train, rec_dictionary=cbf_bpr_dict)


    def recommend(self, target_id, n_tracks=None):
        user_group = self.get_user_group(target_id)
        if user_group <= 4:
            return self.cbf_bpr_rec.recommend(user_id_array=target_id, cutoff=n_tracks)
        else:
            return self.cbf_elastic_rec.recommend(user_id_array=target_id, cutoff=n_tracks)

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

    with open('User-wise_recommendations_cbf_elasticnet_and_cbf_bpr.csv',
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
    provide_recommendations(utility.build_urm_matrix().tocsr())

    """
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)

    cbf_elastic_dict = {}
    elastic = MultiThreadSLIM_ElasticNet(urm_train)
    l1_value = 1e-05
    l2_value = 0.002
    k = 150
    elastic.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)
    cbf = CBF_recommender(urm_csr=urm_train, icm_csr=icm_complete)
    cbf.fit(topK=100, shrink=3)
    cbf_elastic_dict[elastic] = 20.0
    cbf_elastic_dict[cbf] = 1.0
    cbf_elastic_rec = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=cbf_elastic_dict)

    cbf_bpr_dict = {}
    bpr = SLIM_BPR_Cython(urm_train)
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)
    cbf_bpr_dict[bpr] = 1.2
    cbf_bpr_dict[cbf] = 1.0
    cbf_bpr_rec = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=cbf_bpr_dict)

    profile_length = np.ediff1d(urm_train.indptr)
    block_size = int(len(profile_length) * 0.10)
    sorted_users = np.argsort(profile_length)

    MAP_cbf_elastic_per_group = []
    MAP_cbf_bpr_per_group = []
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

        results, _ = evaluator_test.evaluateRecommender(cbf_elastic_rec)
        MAP_cbf_elastic_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(cbf_bpr_rec)
        MAP_cbf_bpr_per_group.append(results[cutoff]["MAP"])

    pyplot.plot(MAP_cbf_bpr_per_group, label="cbf_bpr")
    pyplot.plot(MAP_cbf_elastic_per_group, label="cbf_elastic")
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()
    """
