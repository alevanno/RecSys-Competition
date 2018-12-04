from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Hybrids_recommenders.Experimental_hybrid_recommender import Item_based_CF_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from sklearn.preprocessing import StandardScaler
import numpy as np
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.evaluation_function import evaluate_algorithm

class Item_SLIM_Hybrid_Recommender(object):
    def __init__(self, urm_csr):
        self.urm_csr = urm_csr
        self.item_based_rec = Item_based_CF_recommender(self.urm_csr)
        self.slim_rec = MultiThreadSLIM_ElasticNet(self.urm_csr)

    def fit(self, i=1.0, s=1.0):
        print(s)
        self.item_based_rec.fit(topK=150, shrink=20)
        l1_value = 1e-05
        l2_value = 0.002
        k = 150
        self.slim_rec.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, \
                        l2_penalty=l2_value, topK=k)
        self.W_sparse = i * self.item_based_rec.similarity_csr + s * self.slim_rec.W_sparse.tocsr()


    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        target_profile = self.urm_csr.getrow(target_id)
        scores = target_profile.dot(self.W_sparse).toarray().ravel()
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
    targets_array = utility.get_target_list()
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
    recommender = Item_SLIM_Hybrid_Recommender(urm_csr=urm_train)
    s = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    for value in s:
        recommender.fit(s=value)
        print(evaluate_algorithm(URM_test=urm_test, recommender_object=recommender, at=10))