from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Base.Recommender import Recommender

from lightfm import LightFM
import numpy as np
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score


class LightFM_recommender(Recommender):
    def __init__(self, URM_train, ICM):
        super(LightFM_recommender, self).__init__()
        self.URM_train = URM_train
        self.ICM = ICM

    def fit(self, no_components=0,
                    loss='warp',
                    learning_schedule='adagrad',
                    max_sampled=100,
                    user_alpha=0.0,
                    item_alpha=0.0, epochs=1):
        self.model = LightFM(no_components=no_components,
                    loss='warp',
                    learning_schedule='adagrad',
                    max_sampled=100,
                    user_alpha=user_alpha,
                    item_alpha=item_alpha)

        self.model.fit(interactions=self.URM_train, item_features=self.ICM, epochs=epochs, num_threads=4)


    def compute_item_score(self, user_id_array):
        items_ids = np.array([x for x in range(self.URM_train.shape[1])] * len(user_id_array), dtype='int32')
        user_list = []
        for user in user_id_array:
            user_list.extend([user] * self.URM_train.shape[1])
        user_ids = np.array(user_list, dtype='int32')
        scores = self.model.predict(user_ids=user_ids, item_ids=items_ids, item_features=self.ICM, num_threads=4)

        scores_batch = []
        block = 0
        while block < self.URM_train.shape[1]*len(user_id_array):
            scores_batch.extend([scores[block: block + self.URM_train.shape[1]]])
            block += self.URM_train.shape[1]
        scores = np.array(scores_batch)
        return scores



if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_csr = utility.build_urm_matrix().tocsr()
    icm_csr = utility.build_icm_matrix().tocsr()

    urm_train, urm_test = train_test_holdout(URM_all=urm_csr)
    """
    best = LightFM_recommender(urm_train, icm_csr)
    best.fit(no_components=50, item_alpha=1e-05, user_alpha=0.0001, epochs=90)

    
    print("best so far")
    print(best.evaluateRecommendations(URM_test=urm_test, at=10))


    #alpha_item = [1e-05, 1e-04, 1e-03, 0.01, 0.1, 1e-05, 1e-05, 1e-05, 1e-05, 1e-06, 1e-07, 1e-08]
    #alpha_user = [1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-04, 1e-03, 0.01, 0.1, 1e-05, 1e-05, 1e-05]
    recommender = LightFM_recommender(URM_train=urm_train, ICM=icm_csr)
    recommender.fit(no_components=50, item_alpha=1e-05, user_alpha=0.0001, epochs=90, loss='warp')
    print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
    """
    l_bpr = LightFM_recommender(URM_train=urm_train, ICM=icm_csr)
    l_bpr.fit(no_components=50, item_alpha=1e-05, user_alpha=0.0001, epochs=90, loss='bpr')
    """
    for id in range(urm_train.shape[0]):
        scores = l_bpr.compute_item_score([id])
        print("Minimum: " + str(scores[np.nonzero(scores)].min()))
        print("Maximum: " + str(scores.max()))
        print("mean: " + str(scores[np.nonzero(scores)].mean()))
    print(l_bpr.evaluateRecommendations(URM_test=urm_test, at=10))
    """
    epochs_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250]
    num_components = 50

    recommender = LightFM_recommender(urm_train, icm_csr)

    for epochs in epochs_list:
        print("epochs = " + str(epochs))
        recommender.fit(no_components=num_components, item_alpha=1e-05, user_alpha=0.0001, epochs=epochs,
                        learning_schedule='adadelta')

        print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
