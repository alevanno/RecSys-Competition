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

####################################################################################################################
def provide_recommendations():
    recommendations = {}
    targets_array = utility.get_target_list()

    for target in targets_array:
        recommendations[target] = recommender.recommend(user_id_array=target, cutoff=10)

    with open('Linear_combination_of_elastic_bpr_cbf_graph.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_complete = utility.build_urm_matrix()
    icm_complete = utility.build_icm_matrix()
    #urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    elastic_cbf_dictionary = {}
    bpr_cbf_dictionary = {}

    elastic = MultiThreadSLIM_ElasticNet(urm_complete.tocsr())
    l1_value = 1e-05
    l2_value = 0.002
    k = 150
    elastic.fit(alpha=l1_value+l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)

    bpr = SLIM_BPR_Cython(urm_complete.tocsr())
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

    cbf = CBF_recommender(urm_csr=urm_complete.tocsr(), icm_csr=icm_complete.tocsr())
    cbf.fit(topK=100, shrink=3)
    """
    elastic_cbf_dictionary[cbf] = 1.0
    elastic_cbf_dictionary[elastic] = 20.0
    elastic_cbf = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=elastic_cbf_dictionary)
    print("baseline:")
    print(elastic_cbf.evaluateRecommendations(URM_test=urm_test, at=10))
    

    bpr_cbf_dictionary[cbf] = 1.0
    bpr_cbf_dictionary[bpr] = 1.2
    bpr_cbf = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=bpr_cbf_dictionary)

    recommender_dictionary = {}
    recommender_dictionary[elastic_cbf] = 1.0
    recommender_dictionary[bpr_cbf] = 1.0
    recommender = Linear_combination_scores_recommender(urm_csr=urm_complete, rec_dictionary=recommender_dictionary)
    print("Best so far")
    print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
    
    
    new_dict = {}
    user = User_based_CF_recommender(urm_complete.tocsr())
    user.fit(topK=180, shrink=2)
    
    
    team_bpr_dict = {}
    team_bpr_dict[bpr] = 5.0
    team_bpr_dict[user] = 2.0
    team_bpr_dict[cbf] = 3.0
    team_bpr = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=team_bpr_dict)
    """

    graph = RP3betaRecommender(urm_complete.tocsr())
    graph.fit(topK=100, alpha=0.95, beta=0.3)

    """
    second_level_dict = {}

    second_level_dict[team_bpr] = 1.0
    second_level_dict[elastic_cbf] = 4.0
    second_level_rec = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=second_level_dict)
    print("Second best")
    print(second_level_rec.evaluateRecommendations(URM_test=urm_test, at=10))
    """

    plain_dict = {}

    """
    elastic_list = [50.0]
    bpr_list = [4.5]
    cbf_list = [6.5]
    graph_list = [10.0]
    """

    plain_dict[elastic] = 50.0
    plain_dict[bpr] = 4.5
    plain_dict[cbf] = 6.5
    plain_dict[graph] = 10.0
    recommender = Linear_combination_scores_recommender(urm_csr=urm_complete.tocsr(), rec_dictionary=plain_dict)
    provide_recommendations()


    """
    for index in range(len(elastic_list)):
        print("Conf-> elastic=" + str(elastic_list[index]) + ", bpr=" + str(bpr_list[index]) + ", cbf=" +
              str(cbf_list[index]) + ", graph=" + str(graph_list[index]))
        plain_dict[elastic] = elastic_list[index]
        plain_dict[bpr] = bpr_list[index]
        plain_dict[cbf] = cbf_list[index]
        plain_dict[graph] = graph_list[index]
        new_rec = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=plain_dict)
        print(new_rec.evaluateRecommendations(URM_test=urm_test, at=10))
    """