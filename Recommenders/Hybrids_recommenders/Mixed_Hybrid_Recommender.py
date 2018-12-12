from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Hybrids_recommenders.Slim_and_BPR_CF import SLIM__BPR_and_CF_Hybrid_Recommender


class Mixed_Hybrid_Recommender(object):
    def __init__(self, urm_csr, large_interactions_list):
        #self.urm_csr = urm_csr
        self.slim_rec = MultiThreadSLIM_ElasticNet(URM_train=urm_csr)
        #self.bpr_rec = SLIM_BPR_Cython(URM_train=urm_csr)
        self.hybrid_rec = SLIM__BPR_and_CF_Hybrid_Recommender(urm_csr)
        self.large_interactions_list = large_interactions_list


    def fit(self):
        l1_value = 1e-05
        l2_value = 0.002
        k = 150
        self.slim_rec.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)
        self.hybrid_rec.fit()

    def recommend(self, target_id):
        if target_id in self.large_interactions_list:
            return self.slim_rec.recommend(target_id, cutoff=10)
        else:
            return self.hybrid_rec.recommend(target_id, n_tracks=10)


##################################################################################################################

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_array = utility.get_target_list()

    recommender = Mixed_Hybrid_Recommender(urm_csr=urm_csr, large_interactions_list=utility.user_to_neglect(intensity_interaction='large', n_interactions=30))
    c = 0
    for i in targets_array:
        if i in recommender.large_interactions_list:
            c+=1
    print(c)

    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target)

    with open('MixedRanking_SLIM_and_Hybrid.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))


if __name__ == '__main__':
    utility = Data_matrix_utility()
    provide_recommendations(utility.build_urm_matrix())
