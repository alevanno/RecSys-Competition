from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Base.Evaluation.Evaluator import SequentialEvaluator
from Recommenders.ML_recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.CBF_and_CF_Recommenders.Item_based_CF_Recommender import Item_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender
from Recommenders.Hybrids_recommenders.Hybrids_for_User_wise.ElasticNet_and_BPR_Recommender import ElasticNet_and_BPR_recommender
from Recommenders.Hybrids_recommenders.Hybrids_for_User_wise.ElasticNet_BPR_and_Graph import ElasticNet_BPR_and_Graph
from Recommenders.Hybrids_recommenders.Hybrids_for_User_wise.ElasticNet_and_BPR_Recommender import ElasticNet_and_BPR_recommender
from Recommenders.Hybrids_recommenders.Hybrids_for_User_wise.ElasticNet_and_Graph import ElasticNet_and_Graph
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet

import numpy as np
import matplotlib.pyplot as pyplot

class User_wise_Recommender(object):
    def __init__(self, urm_csr, group_dict):
        self.group_dict = group_dict
        self.bpr_rec = SLIM_BPR_Cython(urm_csr)
        self.ebg_rec = ElasticNet_BPR_and_Graph(urm_csr)
        self.eb_rec = ElasticNet_and_BPR_recommender(urm_csr)
        self.eg_rec = ElasticNet_and_Graph(urm_csr)
        self.slim_rec = MultiThreadSLIM_ElasticNet(urm_csr)

    def fit(self):
        l1_value = 1e-05
        l2_value = 0.002
        k = 150
        self.slim_rec.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)
        self.bpr_rec.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)
        self.ebg_rec.fit()
        self.eb_rec.fit()
        self.eg_rec.fit()

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        user_group = self.get_user_group(target_id)
        if user_group == 0:
            return self.bpr_rec.recommend(user_id_array=target_id, cutoff=n_tracks, remove_seen_flag=exclude_seen)
        elif 1 <= user_group <= 2:
            return self.eb_rec.recommend(target_id=target_id, n_tracks=n_tracks, exclude_seen=exclude_seen)
        elif 3 <= user_group <= 4:
            return self.ebg_rec.recommend(target_id=target_id, n_tracks=n_tracks, exclude_seen=exclude_seen)
        elif user_group == 5:
            return self.eg_rec.recommend(target_id=target_id, n_tracks=n_tracks,exclude_seen=exclude_seen)
        elif user_group == 6:
            return self.slim_rec.recommend(user_id_array=target_id, cutoff=n_tracks, remove_seen_flag=exclude_seen)
        elif user_group == 7:
            return self.eg_rec.recommend(target_id=target_id, n_tracks=n_tracks, exclude_seen=exclude_seen)
        else:
            return self.slim_rec.recommend(user_id_array=target_id, cutoff=n_tracks, remove_seen_flag=exclude_seen)

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
    recommender = User_wise_Recommender(urm_csr=urm_csr, group_dict=group_formation(urm_csr))
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target, n_tracks=10, exclude_seen=True)

    with open('User-wise_recommendations.csv',
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
    provide_recommendations(utility.build_urm_matrix())

    """
    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)

    item_based_rec = Item_based_CF_recommender(urm_train)
    item_based_rec.fit(topK=120, shrink=15)

    user_based_rec = User_based_CF_recommender(urm_train)
    user_based_rec.fit(topK=180, shrink=2)

    slim_rec = MultiThreadSLIM_ElasticNet(urm_train)
    l1_value = 1e-05
    l2_value = 0.002
    k = 150
    slim_rec.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)

    bpr_rec = SLIM_BPR_Cython(urm_train)
    bpr_rec.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

    graph_rec = RP3betaRecommender(urm_train)
    graph_rec.fit(topK=100, alpha=0.95, beta=0.3)

    profile_length = np.ediff1d(urm_train.indptr)
    block_size = int(len(profile_length) * 0.10)
    sorted_users = np.argsort(profile_length)

    MAP_item_per_group = []
    MAP_user_per_group = []
    MAP_slim_per_group = []
    MAP_bpr_per_group = []
    MAP_graph_per_group = []
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

        results, _ = evaluator_test.evaluateRecommender(item_based_rec)
        MAP_item_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(user_based_rec)
        MAP_user_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(slim_rec)
        MAP_slim_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(bpr_rec)
        MAP_bpr_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(graph_rec)
        MAP_graph_per_group.append(results[cutoff]["MAP"])

    pyplot.plot(MAP_item_per_group, label="item")
    pyplot.plot(MAP_user_per_group, label="user")
    pyplot.plot(MAP_slim_per_group, label="slim")
    pyplot.plot(MAP_bpr_per_group, label="bpr")
    pyplot.plot(MAP_graph_per_group, label="graph")
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()
    """


