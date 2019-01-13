from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.Base.Recommender import Recommender

from lightfm import LightFM
import numpy as np

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

    alpha_item = 1e-05
    alpha_user = 1e-05
    epochs = 70
    num_components_list = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80]

    recommender = LightFM_recommender(urm_train, icm_csr)

    for component in num_components_list:
        print("Component = " + str(component))
        recommender.fit(no_components=component, item_alpha=alpha_item, user_alpha=alpha_user, epochs=epochs)
        print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))


