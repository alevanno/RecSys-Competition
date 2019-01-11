from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.CBF_and_CF_Recommenders.Content_based_filtering import CBF_recommender
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender
from Recommenders.Hybrids_recommenders.Linear_combination_scores_recommender import Linear_combination_scores_recommender
from Recommenders.ML_recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.Utilities.Base.Evaluation.Evaluator import SequentialEvaluator

import pandas as pd
import numpy as np
import scipy as sps
import xgboost as xgb
import random

class XGBoost_Recommender(object):

    def __init__(self, recommender, urm_csr, icm_csr):
        self.recommender_to_rerank = recommender #recommender must be already fitted
        self.URM_train = urm_csr
        self.ICM = icm_csr

        self.content_based_rec = CBF_recommender(urm_csr=self.URM_train, icm_csr=self.ICM)
        self.content_based_rec.fit(topK=100, shrink=3)

        self.user_based_rec = User_based_CF_recommender(URM_train=self.URM_train)
        self.user_based_rec.fit(topK=180, shrink=2)

        self.df_with_no_features = self.build_df(self.URM_train)

    def get_URM_train(self):
        return self.URM_train

    def build_features_df(self, df):
        df = self.enrich_df(df, self.content_based_rec, 'cbf')
        df = self.enrich_df(df, self.user_based_rec, 'user_based')

        return self.add_profile_lenght(df).copy()

    def add_profile_lenght(self, df):
        rows = self.URM_train.indptr
        numRatings = np.ediff1d(rows)
        ratings_list = []
        for user in df['playlist_id']:
            ratings_list.append(numRatings[user])
        df['profile_lenght'] = pd.Series(ratings_list, index=df.index)
        del ratings_list
        return df

    def add_icm_features(self, df):

        tracks_df = utility.get_tracks_dataframe()
        max_album_id = tracks_df.sort_values(by=['album_id'], ascending=False)['album_id'].iloc[0]

        album_ids = []
        artist_ids = []
        print("inizio")
        for item in df['track_id']:
            entry = tracks_df[tracks_df['track_id'] == item]
            album_ids.append(entry['album_id'].tolist()[0])
            artist_ids.append(entry['artist_id'].tolist()[0] + max_album_id + 1)
        df['album'] = pd.Series(data=album_ids, index=df.index)
        df['artist'] = pd.Series(data=artist_ids, index=df.index)
        del album_ids
        del artist_ids
        print("finito")
        for album in df.album.unique():
            df[album] = (df.album == album).astype(int)

        df = df.drop(columns=['album'])

        for artist in df.artist.unique():
            df[artist] = (df.artist == artist).astype(int)
        return df.drop(columns=['artist'])

    def build_df(self, urm):
        urm = urm.tocoo()
        user_list = []
        item_list = []

        for user, item in zip(urm.row, urm.col):
            user_list.append(user)
            item_list.append(item)

        dict = {'playlist_id': user_list, 'track_id': item_list}

        return pd.DataFrame(data=dict)

    def enrich_df(self, df, recommender, feature_name):  # recommender must be already fitted
        score = []
        block = 5000
        current_block = 0

        while current_block < df.shape[0]:
            if current_block + block <= df.shape[0]:
                batch_df = df.iloc[current_block : current_block + block]
                current_block += block
            else:
                batch_df = df.iloc[current_block : df.shape[0]]
                current_block = df.shape[0]
            users = batch_df['playlist_id'].drop_duplicates().tolist()
            total_score = recommender.compute_item_score(users)
            for user, item in zip(batch_df['playlist_id'], batch_df['track_id']):
                index = users.index(user)
                score.append(total_score[index, item])

        df[feature_name] = pd.Series(score, index=df.index)
        del score
        return df

    def fit(self, max_depth=3, eta=0.3, silent=1, num_round=20):
        playlist_list = self.df_with_no_features['playlist_id'].tolist()
        track_list = self.df_with_no_features['track_id'].tolist()
        label_list = np.ones((len(playlist_list,))).tolist()
        users_list = self.df_with_no_features['playlist_id'].drop_duplicates().tolist()
        negative_samples = self.compute_negative_samples(users_list)
        print(negative_samples)
        print("I'm here")

        for index in range(len(users_list)):
            playlist_list.extend([users_list[index]] * len(negative_samples[index]))
            track_list.extend(negative_samples[index])
            label_list.extend([0] * len(negative_samples[index]))

        print("negative samples number = ")
        count = 0
        for label in label_list:
            if label == 0:
                count += 1
        print(count)

        dict = {'playlist_id': playlist_list, 'track_id': track_list, 'label': label_list}
        self.features_data = pd.DataFrame(data=dict)
        self.features_data = self.build_features_df(self.features_data)
        self.features_data = self.features_data.sort_values(by=['playlist_id'], ascending=True)

        del playlist_list
        del track_list
        del users_list

        data = self.features_data.drop(columns=['track_id', 'label'])

        dtrain = xgb.DMatrix(data=data, label=self.features_data['label'])
        params = {
            'max_depth': max_depth,  # the maximum depth of each tree
            'eta': eta,  # step for each iteration (shrinkage term to prevent overfitting)
            'silent': silent,  # keep it quiet
            'objective': 'multi:softprob',   # error evaluation for multiclass training
            'num_class': 2,
            'eval_metric': 'merror'}  # evaluation metric

        num_round = num_round  # the number of training iterations (number of trees)

        model = xgb.train(params,
                          dtrain,
                          num_round,
                          verbose_eval=2,
                          evals=[(dtrain, 'train')])
        self.model = model

    def recommend_function(self, user_id_array, intermediate_cutoff=None, final_cutoff=None):
        recommendations = self.recommender_to_rerank.recommend(user_id_array=user_id_array, cutoff=intermediate_cutoff)

        user_list_for_df = []
        item_list_for_df = []
        for user, items in zip(user_id_array, recommendations):
            user_list_for_df.extend([user] * intermediate_cutoff)
            item_list_for_df.extend(items)

        dict = {'playlist_id': user_list_for_df, 'track_id': item_list_for_df}
        test_df = pd.DataFrame(data=dict)
        test_df = self.build_features_df(test_df)
        test_df = test_df.drop(columns=['track_id'])

        dtest = xgb.DMatrix(data=test_df)
        reranked_probabilities = self.model.predict(dtest)

        last_position = 0
        items_recommended = []
        while last_position < len(item_list_for_df):
            items_tuples = []
            items_of_user = recommendations[int(last_position/intermediate_cutoff)]
            probabilities_of_user = reranked_probabilities[last_position:last_position + intermediate_cutoff].tolist()
            probabilities_of_user = [x[1] for x in probabilities_of_user]

            for item, score in zip(items_of_user, probabilities_of_user):
                items_tuples.append((item, score))
            items_tuples = sorted(items_tuples, key=lambda x: x[1], reverse=True)[0 : final_cutoff]
            items_recommended.append([x[0] for x in items_tuples])
            last_position += intermediate_cutoff

        return items_recommended

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False):
        return self.recommend_function(user_id_array=user_id_array, intermediate_cutoff=20, final_cutoff=cutoff)

    def compute_negative_samples(self, user_id_array):
        batch = 1000
        current_position = 0
        negative_samples = []
        while current_position < len(user_id_array):
            if current_position + batch <= len(user_id_array):
                scores = self.recommender_to_rerank.compute_item_score(
                    user_id_array[current_position : current_position + batch])
                current_position += batch
            else:
                scores = self.recommender_to_rerank.compute_item_score(
                    user_id_array[current_position : len(user_id_array)])
                current_position = len(user_id_array)

            for score_list in scores:
                minimum_score = score_list.min()
                negative_list_of_user = []
                for i in range(len(score_list)):
                    if score_list[i] == minimum_score:
                        negative_list_of_user.append(i)

                if len(negative_list_of_user) > 15:
                    temp_list = []
                    random.seed(3)
                    for iteration in range(15):
                        index = random.randint(0, len(negative_list_of_user))
                        temp_list.append(negative_list_of_user[index])
                    negative_list_of_user = set(temp_list)

                negative_samples.append(list(negative_list_of_user))
        return negative_samples



def user_to_compute_for_score(URM_test):
    n_users = URM_test.shape[0]
    usersToEvaluate_mask = np.zeros(n_users, dtype=np.bool)
    rows = URM_test.indptr
    numRatings = np.ediff1d(rows)
    new_mask = numRatings >= 1
    usersToEvaluate_mask = np.logical_or(usersToEvaluate_mask, new_mask)
    usersToEvaluate = np.arange(n_users)[usersToEvaluate_mask]
    return usersToEvaluate


if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_csr = utility.build_urm_matrix().tocsr()
    icm_csr = utility.build_icm_matrix().tocsr()

    urm_train, urm_test = train_test_holdout(URM_all=urm_csr)

    elastic = MultiThreadSLIM_ElasticNet(urm_train)
    l1_value = 1e-05
    l2_value = 0.002
    k = 150
    elastic.fit(alpha=l1_value + l2_value, l1_penalty=l1_value, l2_penalty=l2_value, topK=k)

    bpr = SLIM_BPR_Cython(urm_train)
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

    cbf = CBF_recommender(urm_csr=urm_train, icm_csr=icm_csr)
    cbf.fit(topK=100, shrink=3)

    graph = RP3betaRecommender(urm_train)
    graph.fit(topK=100, alpha=0.95, beta=0.3)

    rec_dictionary = {}
    rec_dictionary[elastic] = 50.0
    rec_dictionary[bpr] = 4.5
    rec_dictionary[cbf] = 6.5
    rec_dictionary[graph] = 10.0

    recommender_to_rerank = Linear_combination_scores_recommender(urm_csr=urm_train, rec_dictionary=rec_dictionary)

    print("Algorithm with no rerank: ")
    print(recommender_to_rerank.evaluateRecommendations(URM_test=urm_test, at=10))
    recommender = XGBoost_Recommender(urm_csr=urm_train, icm_csr=icm_csr, recommender=recommender_to_rerank)
    recommender.fit(num_round=20, eta=0.3, max_depth=3)

    evaluator = SequentialEvaluator(urm_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)
    print("Algorithm with rerank: \n" + str(results[10]))

    """
    depth_list = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 50, 100]
    for depth in depth_list:
        print("depth=" + str(depth))
        recommender.fit(num_round=10, eta=0.01, max_depth=depth)

        evaluator = SequentialEvaluator(urm_test, cutoff_list=[10])
        results, _ = evaluator.evaluateRecommender(recommender)
        print("Algorithm with rerank: \n" + str(results[10]))
    """
