from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Hybrids_recommenders.Experimental_hybrid_recommender import Item_based_CF_recommender
from Recommenders.Hybrids_recommenders.Experimental_hybrid_recommender import User_based_CF_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from sklearn.preprocessing import StandardScaler
import numpy as np
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

class CF_SLIM_Hybrid_Recommender(object):
    def __init__(self, urm_csr, scale=True, i=1.0, u=1.0, s=1.0, b=1.0):
        self.urm_csr = urm_csr
        self.item_based_rec = Item_based_CF_recommender(self.urm_csr)
        self.user_based_rec = User_based_CF_recommender(self.urm_csr)
        self.slim_rec = MultiThreadSLIM_ElasticNet(self.urm_csr)
        self.bpr_rec = SLIM_BPR_Cython(URM_train=self.urm_csr)
        self.scale = scale
        self.i = i
        self.u = u
        self.s = s
        self.b = b

    def fit(self):
        self.item_based_rec.fit(topK=150, shrink=20)
        self.user_based_rec.fit(topK=180, shrink=2)
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
        item_based_scores = self.item_based_rec.recommend(target_id)
        user_based_scores = self.user_based_rec.recommend(target_id)
        slim_scores = np.ravel(self.slim_rec.compute_item_score(target_id))
        bpr_scores = np.ravel(self.bpr_rec.compute_item_score(target_id))
        #print("Item based not standard: " + str(item_based_scores.mean()))
        #print("User based not standard: " + str(user_based_scores.mean()))
        #print("Slim not standard: " + str(slim_scores.mean()))

        if self.scale:
            item_based_std_scores = self.standardize(item_based_scores)
            user_based_std_scores = self.standardize(user_based_scores)
            slim_std_scores = self.standardize(slim_scores)
            bpr_std_scores = self.standardize(bpr_scores)
            #print("Item based standard: " + str(item_based_std_scores.mean()))
            #print("User based standard: " + str(user_based_std_scores.mean()))
            #print("Slim standard: " + str(slim_std_scores.mean()))
            scores = self.i * item_based_std_scores + self.u * user_based_std_scores + self.s * slim_std_scores\
                    + self.b * bpr_std_scores
        else:
            scores = self.i * item_based_scores + self.u * user_based_scores + self.s * slim_scores \
                     + self.b * bpr_scores

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
    recommender = CF_SLIM_Hybrid_Recommender(urm_csr=urm_csr, scale=True)
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target, n_tracks=10)

    with open('SLIM_and_BPR_Item_User_CF_Hybrid_scores_sum_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
    """
    provide_recommendations(utility.build_urm_matrix())
    """

    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    recommender_list = (Item_based_CF_recommender(urm_train), User_based_CF_recommender(urm_train), MultiThreadSLIM_ElasticNet(urm_train), SLIM_BPR_Cython(URM_train=urm_train))
    for recommender in recommender_list:
        recommender.fit()
        print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
    print(evaluate_algorithm(CF_SLIM_Hybrid_Recommender(urm_csr=urm_train, scale=True)))
