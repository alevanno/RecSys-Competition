from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Hybrids_recommenders.Experimental_hybrid_recommender import Item_based_CF_recommender
from Recommenders.Hybrids_recommenders.Experimental_hybrid_recommender import User_based_CF_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from sklearn.preprocessing import StandardScaler
import numpy as np
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ML_recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.CBF_and_CF_Recommenders.Content_based_filtering import CBF_recommender
from Recommenders.Hybrids_recommenders.Hybrids_for_User_wise.User_and_item_CF import User_and_item_CF
from Recommenders.Hybrids_recommenders.SLIM__BPR_Graph_and_CF_Hybrid_Recommender import SLIM__BPR_Graph_and_CF_Hybrid_Recommender

class ML_and_not_recommender(object):
    def __init__(self, urm_csr, icm_csr, scale=True, n=1.0, s=1.0, b=1.0, g=1.0):
        self.urm_csr = urm_csr
        self.icm_csr = icm_csr
        self.non_ml = User_and_item_CF(urm_csr=self.urm_csr, icm_csr=self.icm_csr, i=5.0, c=1.4, u=1.0, scale=False)
        self.slim_rec = MultiThreadSLIM_ElasticNet(self.urm_csr)
        self.bpr_rec = SLIM_BPR_Cython(URM_train=self.urm_csr)
        self.graph_rec = RP3betaRecommender(self.urm_csr)
        self.scale = scale
        self.n = n
        self.s = s
        self.b = b
        self.g = g

    def fit(self):
        self.non_ml.fit()
        l1_value = 1e-05
        l2_value = 0.002
        k = 150
        self.slim_rec.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)
        self.bpr_rec.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)
        self.graph_rec.fit(topK=100, alpha=0.95, beta=0.3)


    def standardize(self, array):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel()

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        non_ml_scores = self.non_ml.get_scores(target_id=target_id)
        slim_scores = np.ravel(self.slim_rec.compute_item_score(target_id))
        bpr_scores = np.ravel(self.bpr_rec.compute_item_score(target_id))
        graph_based_scores = np.ravel(self.graph_rec.compute_item_score(target_id))

        if self.scale:
            non_ml_std_scores = self.standardize(non_ml_scores)
            slim_std_scores = self.standardize(slim_scores)
            bpr_std_scores = self.standardize(bpr_scores)
            graph_based_std_scores = self.standardize(graph_based_scores)
            scores = self.g * graph_based_std_scores + self.s * slim_std_scores + self.b * bpr_std_scores + \
                        self.n * non_ml_std_scores
        else:
            scores = self.g * graph_based_scores + self.s * slim_scores + self.b * bpr_scores + \
                        self.n * non_ml_scores


        if exclude_seen:
            scores = self.filter_seen(target_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:n_tracks]

    def filter_seen(self, target_id, scores):
        start_pos = self.urm_csr.indptr[target_id] #extracts the column in which the target start
        end_pos = self.urm_csr.indptr[target_id+1] #extracts the column in which the target ends
        target_profile = self.urm_csr.indices[start_pos:end_pos] #extracts the columns indexes
                                #in which we can find the non zero values in target
        scores[target_profile] = -np.inf
        return scores
    #############################################################################################

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_array = utility.get_target_list()
    #recommender = ML_and_not_recommender(urm_csr=urm_csr, icm_csr=icm_complete, s=50.0, b=2.0, g=0.0, n=1.0, scale=False)
    recommender = SLIM__BPR_Graph_and_CF_Hybrid_Recommender(i=1.0, u=1.0, g=0.0, s=1.0, b=1.0, c=0.0, scale=True)
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target, n_tracks=10)

    with open('ML_and_not_recommendations_standard_with_parameters.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))


def old_experiment(i, u, s, b, g, c, scale, urm_validation, exclude_seen=True):
    print("Good Configuration -> i=" + str(i) + ", u=" + str(u) + ", s=" + str(s) + ", b=" + str(b) + ", g=" + str(g) + ", c=" + str(c) + ", scale=" + str(scale))
    recommender = SLIM__BPR_Graph_and_CF_Hybrid_Recommender(urm_csr=urm_train, icm_csr=icm_complete, scale=scale, i=i, u=u, s=s, b=b, g=g)
    recommender.fit()
    print(evaluate_algorithm(URM_test=urm_validation, recommender_object=recommender, at=10, exclude_seen=exclude_seen))


def new_experiment(s, b, g, n, scale, urm_validation, exclude_seen=True):
    print("Configuration -> s=" + str(s) + ", b=" + str(b) + ", g=" + str(g) + ", n=" + str(n) + ", scale=" + str(scale))
    recommender = ML_and_not_recommender(urm_csr=urm_train, icm_csr=icm_complete, scale=scale, s=s, b=b, g=g, n=n)
    recommender.fit()
    print(evaluate_algorithm(URM_test=urm_validation, recommender_object=recommender, at=10, exclude_seen=exclude_seen))


if __name__ == '__main__':
    utility = Data_matrix_utility()
    #icm_complete = utility.build_icm_matrix()
    #provide_recommendations(utility.build_urm_matrix())


    urm_complete = utility.build_urm_matrix()
    icm_complete = utility.build_icm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    old_experiment(i=1.0, u=1.0, g=0.0, s=1.0, b=1.0, c=0.0, scale=True, urm_validation=urm_test, exclude_seen=True)

    """
    print("Attempt: ")
    new_experiment(s=1, b=1, g=0, n=1, scale=True, urm_validation=urm_test, exclude_seen=True)

    #best s: s=50
    #best b: b=2
    s_list = [50.0, 50.0, 40.0]
    n_list = [1.0, 2.0, 2.0]
    b_list = [2.0, 1.0, 4.0]
    g_list = [0.0, 0.0, 0.0]
    #b_list = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    #g_list = [0.0, 0.5, 1, 2, 4, 6]


    for ex in range(len(s_list)):
        new_experiment(n=n_list[ex], g=g_list[ex], b=b_list[ex], s=s_list[ex], scale=False, urm_validation=urm_test, \
                   exclude_seen=True)
    """