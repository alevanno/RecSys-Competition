from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Hybrids_recommenders.Experimental_hybrid_recommender import Item_based_CF_recommender
from Recommenders.Hybrids_recommenders.Experimental_hybrid_recommender import User_based_CF_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from sklearn.preprocessing import StandardScaler
import numpy as np
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.evaluation_function import evaluate_algorithm

class CF_SLIM_Hybrid_Recommender(object):
    def __init__(self, urm_csr):
        self.urm_csr = urm_csr
        self.item_based_rec = Item_based_CF_recommender(self.urm_csr)
        self.user_based_rec = User_based_CF_recommender(self.urm_csr)
        self.slim_rec = MultiThreadSLIM_ElasticNet(self.urm_csr)

    def fit(self):
        self.item_based_rec.fit(topK=150, shrink=20)
        self.user_based_rec.fit(topK=180, shrink=2)
        l1_value = 1e-05
        l2_value = 0.002
        k = 150
        self.slim_rec.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, \
                        l2_penalty=l2_value, topK=k)

    def standardize(self, array):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel()

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        item_based_scores = self.item_based_rec.recommend(target_id)
        user_based_scores = self.user_based_rec.recommend(target_id)
        slim_scores = self.slim_rec.compute_score_item_based(target_id)
        print("Item based not standard: " + str(item_based_scores.mean()))
        print("User based not standard: " + str(user_based_scores.mean()))
        print("Slim not standard: " + str(slim_scores.mean()))

        item_based_std_scores = self.standardize(item_based_scores)
        user_based_std_scores = self.standardize(user_based_scores)
        slim_std_scores = self.standardize(slim_scores)
        print("Item based standard: " + str(item_based_std_scores.mean()))
        print("User based standard: " + str(user_based_std_scores.mean()))
        print("Slim standard: " + str(slim_std_scores.mean()))

        scores = item_based_std_scores + user_based_std_scores + slim_std_scores
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
    recommender = CF_SLIM_Hybrid_Recommender(urm_csr)
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target, n_tracks=10)

    with open('CF_SLIM_Hybrid_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
    #provide_recommendations(utility.build_urm_matrix())

    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    recommender = CF_SLIM_Hybrid_Recommender(urm_train)
    recommender.fit()

    print(evaluate_algorithm(URM_test=urm_test, recommender_object=recommender, at=10))

#MAP = 0.1146 in evaluation split