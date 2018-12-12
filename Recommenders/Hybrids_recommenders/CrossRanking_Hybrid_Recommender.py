from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.evaluation_function import evaluate_algorithm

class CrossRanking_Hybrid_Recommender(object):
    def __init__(self, urm_csr):
        self.urm_csr = urm_csr
        self.slim_rec = MultiThreadSLIM_ElasticNet(URM_train=urm_csr)
        self.bpr_rec = SLIM_BPR_Cython(URM_train=urm_csr)

    def fit(self):
        l1_value = 1e-05
        l2_value = 0.002
        k = 150
        self.slim_rec.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)
        self.bpr_rec.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

    def recommend(self, target_id, n_tracks=None, n_bpr_items=1):
        slim_recommendation_list = self.slim_rec.recommend(user_id_array=target_id, cutoff=n_tracks)
        bpr_recommendation_list = self.bpr_rec.recommend(user_id_array=target_id, cutoff=n_tracks)

        hybrid_recommendation_list = bpr_recommendation_list[:n_bpr_items]

        for item in slim_recommendation_list:
            if item not in hybrid_recommendation_list:
                hybrid_recommendation_list.append(item)

        return hybrid_recommendation_list[:n_tracks]

##################################################################################################################

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_array = utility.get_target_list()
    recommender = CrossRanking_Hybrid_Recommender(urm_csr=urm_csr)
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target, n_tracks=10, n_bpr_items=3)

    with open('CrossRanking_SLIM_and_BPR.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
    provide_recommendations(utility.build_urm_matrix())

    """
    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    recommender = CrossRanking_Hybrid_Recommender(urm_csr=urm_train)
    recommender.fit()
    n_bpr_items_list = [x for x in range(1, 10)]
    
    for n in n_bpr_items_list:
        print("Evaluate n_bpr_items= " + str(n))
        print(evaluate_algorithm(URM_test=urm_test, recommender_object=recommender, at=10, n_bpr_items=n))
    """