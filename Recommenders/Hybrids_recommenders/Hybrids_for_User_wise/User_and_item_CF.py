from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Hybrids_recommenders.Experimental_hybrid_recommender import Item_based_CF_recommender
from Recommenders.Hybrids_recommenders.Experimental_hybrid_recommender import User_based_CF_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from sklearn.preprocessing import StandardScaler
import numpy as np
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.CBF_and_CF_Recommenders.Content_based_filtering import CBF_recommender

class User_and_item_CF(object):
    def __init__(self, urm_csr, icm_csr, scale=True, i=1.0, u=1.0, c=1.0):
        super(User_and_item_CF, self).__init__()
        self.URM_train = urm_csr
        self.icm_csr = icm_csr
        self.item_based_rec = Item_based_CF_recommender(self.URM_train)
        self.user_based_rec = User_based_CF_recommender(self.URM_train)
        self.cbf = CBF_recommender(self.URM_train, self.icm_csr)
        self.scale = scale
        self.i = i
        self.u = u
        self.c = c

    def fit(self):
        self.item_based_rec.fit(topK=150, shrink=20)
        self.user_based_rec.fit(topK=180, shrink=2)
        self.cbf.fit(topK=180, shrink=2)

    def standardize(self, array):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel()

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        item_based_scores = np.ravel(self.item_based_rec.compute_item_score(target_id))
        user_based_scores = np.ravel(self.user_based_rec.compute_score_user_based(target_id))
        cbf_scores = self.cbf.get_score(target_id)
        #print("Item based not standard: " + str(item_based_scores[np.nonzero(item_based_scores)].mean()))
        #print("User based not standard: " + str(user_based_scores[np.nonzero(user_based_scores)].mean()))

        if self.scale:
            item_based_std_scores = self.standardize(item_based_scores)
            user_based_std_scores = self.standardize(user_based_scores)
            #print("Item based standard: " + str(item_based_std_scores.mean()))
            #print("User based standard: " + str(user_based_std_scores.mean()))
            scores = self.i * item_based_std_scores + self.u * user_based_std_scores
        else:
            scores = self.i * item_based_scores + self.u * user_based_scores + self.c * cbf_scores

        if exclude_seen:
            scores = self.filter_seen(target_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:n_tracks]

    def get_scores(self, target_id):
        item_based_scores = np.ravel(self.item_based_rec.compute_item_score(target_id))
        user_based_scores = np.ravel(self.user_based_rec.compute_score_user_based(target_id))
        cbf_scores = self.cbf.get_score(target_id)
        return self.i * item_based_scores + self.u * user_based_scores + self.c * cbf_scores

    """
    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False):
        ranking = []
        for id in user_id_array:
            ranking.append(self.recommend_user(target_id=id, n_tracks=cutoff, exclude_seen=remove_seen_flag))

        return ranking
    """

    def filter_seen(self, target_id, scores):
        start_pos = self.URM_train.indptr[target_id] #extracts the column in which the target start
        end_pos = self.URM_train.indptr[target_id+1] #extracts the column in which the target ends
        target_profile = self.URM_train.indices[start_pos:end_pos] #extracts the columns indexes
                                #in which we can find the non zero values in target
        scores[target_profile] = -np.inf
        return scores
    #############################################################################################

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_array = utility.get_target_list()
    recommender = User_and_item_CF(urm_csr=urm_csr, scale=True)
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target, n_tracks=10)

    with open('SLIM_and_BPR_Item_User_CF_Hybrid_scores_sum_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
    #provide_recommendations(utility.build_urm_matrix())

    urm_complete = utility.build_urm_matrix()
    icm = utility.build_icm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    print("Item based score")
    item = Item_based_CF_recommender(urm_train)
    item.fit()
    print(item.evaluateRecommendations(URM_test=urm_test, at=10))

    recommender_std = User_and_item_CF(urm_csr=urm_train,icm_csr=icm, scale=True)
    recommender_std.fit()
    print("best score:")
    print(evaluate_algorithm(URM_test=urm_test, recommender_object=recommender_std, at=10))

    c_list = [1.4]
    u_list = [1.0]
    for e in range(len(c_list)):
        print("c=" + str(c_list[e]) + ", u=" + str(u_list[e]))
        recommender = User_and_item_CF(urm_csr=urm_train, scale=False, icm_csr=icm, i=5.0, u=u_list[e], c=c_list[e])
        recommender.fit()
        print(evaluate_algorithm(URM_test=urm_test, recommender_object=recommender, at=10))

#Parametri migliori: i=5.0, c=1.4, u=1.0 con 0.1140 di MAP