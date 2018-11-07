import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix
from User_based_CF_Recommender import CF_recommender
from data_splitter import train_test_holdout

train_path = Path("data")/"train.csv"
target_path = Path('data')/'target_playlists.csv'


class Data_matrix_utility(object):
    def __init__(self, path):
        self.train_path = path

    def build_matrix(self):  # for now it works only for URM
        data = pd.read_csv(self.train_path)
        n_playlists = data.nunique().get('playlist_id')
        n_tracks = data.nunique().get('track_id')

        playlists_array = self.extract_array_from_dataFrame(data, ['track_id'])
        track_array = self.extract_array_from_dataFrame(data, ['playlist_id'])
        implicit_rating = np.ones_like(np.arange(len(track_array)))
        urm = coo_matrix((implicit_rating, (playlists_array, track_array)),
                         shape=(n_playlists, n_tracks))
        return urm

    def extract_array_from_dataFrame(self, data, columns_list_to_drop):
        array = data.drop(columns=columns_list_to_drop).get_values()
        return array.T.squeeze()  # transform a nested array in array and transpose it


def MAP(recommended_items, relevant_items):

    is_relevant = np.in1d(
        recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * \
        np.cumsum(is_relevant, dtype=np.float32) / \
        (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / \
        np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate_algorithm(URM_test, recommender_object, at=10):

    '''cumulative_precision = 0.0
    cumulative_recall = 0.0'''
    cumulative_MAP = 0.0

    num_eval = 0

    print("Starting evaluation on #items " + str(URM_test.shape[0]))

    for play_id in range(URM_test.shape[0]):

        if (play_id % 5000 == 0):
            print("Evaluating item " + str(play_id))

        relevant_items = URM_test[play_id].indices

        if len(relevant_items) > 0:

            recommended_items = recommender_object.recommend(
                target_id=play_id, n_tracks=10)
            num_eval += 1

            '''cumulative_precision += precision(recommended_items,
                                              relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)'''
            cumulative_MAP += MAP(recommended_items, relevant_items)

    '''cumulative_precision /= num_eval
    cumulative_recall /= num_eval'''
    cumulative_MAP /= num_eval

    print("Recommender performance is: MAP = {:.4f}".format(
        cumulative_MAP))

if __name__ == '__main__':
    utility = Data_matrix_utility(train_path)
    urm = utility.build_matrix()
    print("URM_all shape is " + str(urm.shape) + ", with #ratings " + str(urm.nnz))
    (urm_train, urm_test) = train_test_holdout(urm)
    print("URM_train #ratings is " + str(urm_train.nnz))
    print("URM_test #ratings is " + str(urm_test.nnz))
    urm_train_csr = urm_train.tocsr()
    
    for i in range(0,7):
        recommender = CF_recommender(urm_train_csr)
        print("Shrink = " +str(i))
        recommender.fit(shrink=i)
        evaluate_algorithm(urm_test, recommender)
    
    k= 50
    while k<=85:
        recommender = CF_recommender(urm_train_csr)
        print("k = " + str(k))
        recommender.fit(topK=k, shrink= 0)
        evaluate_algorithm(urm_test, recommender)
        k += 5
