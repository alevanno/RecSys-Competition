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

class ElasticNet_and_BPR_recommender(object):
    def __init__(self, urm_csr, scale=True, s=1.0, b=1.0):
        self.urm_csr = urm_csr
        self.slim_rec = MultiThreadSLIM_ElasticNet(self.urm_csr)
        self.bpr_rec = SLIM_BPR_Cython(URM_train=self.urm_csr)
        self.scale = scale
        self.s = s
        self.b = b

    def fit(self):
        l1_value = 1e-05
        l2_value = 0.002
        k = 150
        self.slim_rec.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)
        self.bpr_rec.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)


    def standardize(self, array):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel()

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        slim_scores = np.ravel(self.slim_rec.compute_item_score(target_id))
        bpr_scores = np.ravel(self.bpr_rec.compute_item_score(target_id))

        if self.scale:
            slim_std_scores = self.standardize(slim_scores)
            bpr_std_scores = self.standardize(bpr_scores)
            scores = self.s * slim_std_scores + self.b * bpr_std_scores
        else:
            scores = self.s * slim_scores + self.b * bpr_scores

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

def experiment(s, b, scale, urm_validation, exclude_seen=True):
    print("Configuration -> s=" + str(s) + ", b=" + str(b) + ", scale=" + str(scale))
    recommender = ElasticNet_and_BPR_recommender(urm_csr=urm_train, scale=scale, s=s, b=b)
    recommender.fit()
    print(evaluate_algorithm(URM_test=urm_validation, recommender_object=recommender, at=10, exclude_seen=exclude_seen))


if __name__ == '__main__':
    utility = Data_matrix_utility()
    #provide_recommendations(utility.build_urm_matrix())


    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)

    print("Best experiment")
    experiment(s=1.0, b=1.0, scale=True, urm_validation=urm_test)

    experiment(s=1.0, b=1.0, scale=False, urm_validation=urm_test)
    experiment(s=0.5, b=0.5, scale=False, urm_validation=urm_test)
    experiment(s=0.6, b=0.4, scale=False, urm_validation=urm_test)
    experiment(s=1.5, b=1.0, scale=False, urm_validation=urm_test)