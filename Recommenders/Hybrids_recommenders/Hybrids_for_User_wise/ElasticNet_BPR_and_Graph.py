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

class ElasticNet_BPR_and_Graph(object):
    def __init__(self, urm_csr, scale=True, s=1.0, b=1.0, g=1.0):
        self.urm_csr = urm_csr
        self.slim_rec = MultiThreadSLIM_ElasticNet(self.urm_csr)
        self.bpr_rec = SLIM_BPR_Cython(URM_train=self.urm_csr)
        self.graph_rec = RP3betaRecommender(self.urm_csr)
        self.scale = scale
        self.s = s
        self.b = b
        self.g = g

    def fit(self):
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
        slim_scores = np.ravel(self.slim_rec.compute_item_score(target_id))
        bpr_scores = np.ravel(self.bpr_rec.compute_item_score(target_id))
        graph_based_scores = np.ravel(self.graph_rec.compute_item_score(target_id))

        if self.scale:
            slim_std_scores = self.standardize(slim_scores)
            bpr_std_scores = self.standardize(bpr_scores)
            graph_based_std_scores = self.standardize(graph_based_scores)
            scores = self.g * graph_based_std_scores + self.s * slim_std_scores + self.b * bpr_std_scores
        else:
            scores =  self.g * graph_based_scores + self.s * slim_scores + self.b * bpr_scores

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