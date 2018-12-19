import numpy as np
from sklearn.preprocessing import StandardScaler
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Utilities.data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Compute_Similarity_Python import Compute_Similarity_Python
from Recommenders.CBF_and_CF_Recommenders.Item_based_CF_Recommender import Item_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender


################################################################################
class User_Item_CF_Hybrid_recommender():
    def __init__(self, urm_csr, i, u):
        self.urm_csr = urm_csr
        self.item_based_rec = Item_based_CF_recommender(self.urm_csr)
        self.user_based_rec = User_based_CF_recommender(self.urm_csr)
        self.i = i
        self.u = u

    def fit(self):
        self.item_based_rec.fit(topK=150, shrink=20)
        self.user_based_rec.fit(topK=180, shrink=2)

    def standardize(self, array):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel()

    def recommend(self, user_id, n_tracks=None, remove_seen_flag=True):
        item_based_scores = np.ravel(self.item_based_rec.compute_item_score(user_id))
        user_based_scores = np.ravel(self.user_based_rec.compute_score_user_based(user_id))
        item_based_std_scores = self.standardize(item_based_scores)
        user_based_std_scores = self.standardize(user_based_scores)
        scores = np.add(self.i*item_based_std_scores, self.u*user_based_std_scores)
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

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_array = utility.get_target_list()
    recommender = User_Item_CF_Hybrid_recommender(urm_csr, 0.6, 0.4)
    recommender.fit()
    for target in targets_array:
        recommendations[target] = recommender.recommend(user_id=target, n_tracks=10)

    with open('weighted_experimental_hybrid_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))

if __name__ == '__main__':
    utility = Data_matrix_utility()
   # provide_recommendations(utility.build_urm_matrix())

    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)

    recommender = User_Item_CF_Hybrid_recommender(urm_train, 0.6, 0.4)
    recommender.fit()
    print(evaluate_algorithm(URM_test=urm_test, recommender_object=recommender, at=10))

"""
    i= []
    for value in i:
        recommender = Hybrid_recommender(urm_train, 1, value)
        print("Evaluating: i = " + str(1) + "; u = " + str(value))
        recommender.fit()
        evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender, at=10)
        print(evaluation_metrics)
        #K_results.append(evaluation_metrics["MAP"])

    shrink_value = [x for x in range(10)]
    shrink_result = []
    for value in shrink_value:
        print('Evaluating shrink = ' + str(value))
        recommender.fit(topK = 100,shrink=value)
        evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
        print(evaluation_metrics)
        #shrink_results.append(evaluation_metrics["MAP"])

    #pyplot.plot(K_values, K_results)
    pyplot.plot(shrink_values, shrink_results)
    pyplot.ylabel('MAP')
    pyplot.xlabel('TopK')
    pyplot.show()
    """

"""
Result on public test set: 0.08637
"""
