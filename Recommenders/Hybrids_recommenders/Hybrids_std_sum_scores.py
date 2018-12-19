import numpy as np
from sklearn.preprocessing import StandardScaler
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Utilities.data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Compute_Similarity_Python import Compute_Similarity_Python
from Recommenders.CBF_and_CF_Recommenders.Item_based_CF_Recommender import Item_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


################################################################################
class Hybrids_std_sum_scores(object):
    def __init__(self, urm_csr):
        self.urm_csr = urm_csr
        self.item_based_rec = Item_based_CF_recommender(self.urm_csr)
        self.user_based_rec = User_based_CF_recommender(self.urm_csr)
        self.bpr_rec = SLIM_BPR_Cython(self.urm_csr)


    def fit(self):
        self.item_based_rec.fit(topK=150, shrink=20)
        self.user_based_rec.fit(topK=180, shrink=2)
        self.bpr_rec.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

    def standardize(self, array):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel()

    def recommend(self, user_id, n_tracks=None, remove_seen_flag=True):
        item_based_scores = np.ravel(self.item_based_rec.compute_item_score(user_id))
        user_based_scores = np.ravel(self.user_based_rec.compute_score_user_based(user_id))
        bpr_scores = np.ravel(self.bpr_rec.compute_item_score(user_id))
        item_based_std_scores = self.standardize(item_based_scores)
        user_based_std_scores = self.standardize(user_based_scores)
        bpr_std_scores = self.standardize(bpr_scores)
        scores = item_based_std_scores + user_based_std_scores + bpr_std_scores
        if remove_seen_flag:
            scores = self.filter_seen(user_id, scores)

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
##############################################################################

if __name__ == '__main__':
    utility = Data_matrix_utility()

    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)

    print("Item based CF recommendations")
    item = Item_based_CF_recommender(urm_train)
    item.fit(topK=150, shrink=20)
    print(item.evaluateRecommendations(URM_test=urm_test, at=10))

    print("User based CF recommendations")
    user = User_based_CF_recommender(urm_train)
    user.fit(topK=180, shrink=2)
    print(user.evaluateRecommendations(URM_test=urm_test, at=10))

    print("Hybrid recommendations")
    recommender = Mixed_Hybrid_recommender(urm_train)
    recommender.fit()
    print(evaluate_algorithm(URM_test=urm_test, recommender_object=recommender, at=10))
