from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.CBF_and_CF_Recommenders.Item_based_CF_Recommender import Item_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender
from Recommenders.CBF_and_CF_Recommenders.Content_based_filtering import CBF_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ML_recommenders.FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from Recommenders.CBF_and_CF_Recommenders.User_based_CF_Recommender import User_based_CF_recommender
from Recommenders.ML_recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.ParameterTuning.BayesianSkoptSearch import BayesianSkoptSearch
from Recommenders.ParameterTuning.AbstractClassSearch import DictionaryKeys
from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.Base.Evaluation.Evaluator import SequentialEvaluator
from Recommenders.Hybrids_recommenders.LightFM_recommender import LightFM_recommender
from Recommenders.CBF_and_CF_Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.CBF_and_CF_Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender

import traceback, pickle
import multiprocessing
from multiprocessing import Pool
from functools import partial

import numpy as np
from skopt.space import Real, Integer, Categorical
from sklearn.preprocessing import StandardScaler
import os

class Linear_combination_scores_with_Bayesian_search(Recommender):
    def __init__(self, urm_csr, rec_dictionary, scale=False):
        super(Linear_combination_scores_with_Bayesian_search, self).__init__()
        self.URM_train = urm_csr
        self.rec_dict = rec_dictionary
        self.scale = scale

    def compute_item_score(self, user_id):
        scores = np.zeros(shape=(len(user_id), self.URM_train.shape[1]))

        for tuple in self.rec_dict.items():
            if self.scale:
                shape = (len(user_id), self.URM_train.shape[1])
                scores += self.standardize(tuple[0].compute_item_score(user_id), shape) * tuple[1]
            else:
                scores += tuple[0].compute_item_score(user_id) * tuple[1]

        return scores

    def standardize(self, array, shape):
        array = array.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(array)
        return scaler.transform(array).ravel().reshape(shape)

    def fit(self, elastic_value, bpr_value, graph_value, item_based_value, user_based_value):
        self.rec_dict.clear()
        self.rec_dict[elastic_hybrid] = elastic_value
        self.rec_dict[bpr_hybrid] = bpr_value
        self.rec_dict[graph_hybrid] = graph_value
        self.rec_dict[item_hybrid] = item_based_value
        self.rec_dict[user_based] = user_based_value

    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores

    def saveModel(self, folder_path, file_name = None):
        """
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = self.rec_dict


        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)


        print("{}: Saving complete".format(self.RECOMMENDER_NAME))
        """


def runParameterSearch_Collaborative(recommender_class, URM_train, metric_to_optimize = "PRECISION",
                                     evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                                     output_folder_path ="result_experiments/", parallelizeKNN = True, n_cases = 30):


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME

        parameterSearch = BayesianSkoptSearch(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

        if recommender_class is Linear_combination_scores_with_Bayesian_search:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["elastic_value"] = Real(low=50.0, high=80.0, prior='uniform')
            hyperparamethers_range_dictionary["bpr_value"] = Real(low=0.0, high=10.0, prior='uniform')
            hyperparamethers_range_dictionary["graph_value"] = Real(low=0.0, high=15.0, prior='uniform')
            hyperparamethers_range_dictionary["item_based_value"] = Real(low=0.0, high=10.0, prior='uniform')
            hyperparamethers_range_dictionary["user_based_value"] = Real(low=0.0, high=7.0, prior='uniform')



            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [urm_train, dict()],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        best_parameters = parameterSearch.search(recommenderDictionary,
                                                 n_cases=n_cases,
                                                 output_folder_path=output_folder_path,
                                                 output_file_name_root=output_file_name_root,
                                                 metric_to_optimize=metric_to_optimize)
    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()



if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_complete = utility.build_urm_matrix()
    icm_complete = utility.build_icm_matrix()
    urm_temp_train, urm_test = train_test_holdout(URM_all=urm_complete)

    urm_train, urm_validation = train_test_holdout(URM_all=urm_temp_train)

    elastic_new = MultiThreadSLIM_ElasticNet(urm_train)
    elastic_new.fit(alpha=0.0008868749995645901, l1_penalty=1.8986406043137196e-06,
                    l2_penalty=0.011673969837199876, topK=200)

    cbf_new = ItemKNNCBFRecommender(ICM=icm_complete, URM_train=urm_train)
    cbf_new.fit(topK=50, shrink=100, feature_weighting="TF-IDF")

    item_based = Item_based_CF_recommender(urm_train)
    item_based.fit(topK=150, shrink=20)

    user_based = User_based_CF_recommender(urm_train)
    user_based.fit(topK=180, shrink=2)

    graph = RP3betaRecommender(urm_train)
    graph.fit(topK=100, alpha=0.95, beta=0.3)

    bpr = SLIM_BPR_Cython(urm_train)
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)

    elastic_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_train, Similarity_1=elastic_new.W_sparse,
                                                Similarity_2=cbf_new.W_sparse)
    elastic_hybrid.fit(alpha=0.95, beta=0.05, topK=250)

    item_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_train, Similarity_1=item_based.W_sparse,
                                                Similarity_2=cbf_new.W_sparse)
    item_hybrid.fit(alpha=0.8, beta=0.2, topK=150)

    graph_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_train, Similarity_1=graph.W_sparse,
                                                Similarity_2=cbf_new.W_sparse)
    graph_hybrid.fit(alpha=0.97, beta=0.03, topK=200)

    bpr_hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_train, Similarity_1=bpr.W_sparse,
                                                Similarity_2=cbf_new.W_sparse)
    bpr_hybrid.fit(alpha=0.55, beta=0.45, topK=300)

    ########################################################################################################
    plain_dict = {}
    plain_dict[elastic_new] = 50.0
    plain_dict[bpr] = 4.5
    plain_dict[cbf_new] = 6.5
    plain_dict[graph] = 10.0
    recommender = Linear_combination_scores_with_Bayesian_search(urm_csr=urm_train, rec_dictionary=plain_dict)
    print("Best so far:")
    print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))

    plain_dict.clear()
    plain_dict[elastic_hybrid] = 65.0
    plain_dict[bpr_hybrid] = 9.0
    plain_dict[graph_hybrid] = 12.0
    plain_dict[item_hybrid] = 6.5
    recommender_hybrid = Linear_combination_scores_with_Bayesian_search(urm_csr=urm_train, rec_dictionary=plain_dict)
    print("With hybrid and different parameters:")
    print(recommender_hybrid.evaluateRecommendations(URM_test=urm_test, at=10))

    ########################################################################################################


    #################################################################
    output_folder_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    collaborative_algorithm_list = [
        Linear_combination_scores_with_Bayesian_search
    ]

    evaluator_validation = SequentialEvaluator(urm_validation, cutoff_list=[5])
    evaluator_test = SequentialEvaluator(urm_test, cutoff_list=[5, 10])

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train=urm_train,
                                                       metric_to_optimize="MAP",
                                                       n_cases=30,
                                                       evaluator_validation_earlystopping=evaluator_validation,
                                                       evaluator_validation=evaluator_validation,
                                                       evaluator_test=evaluator_test,
                                                       output_folder_path=output_folder_path)

    for recommender_class in collaborative_algorithm_list:

        try:

            runParameterSearch_Collaborative_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()
