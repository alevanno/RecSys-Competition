#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Utilities.Base.NonPersonalizedRecommender import TopPop, Random
from Recommenders.CBF_and_CF_Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.CBF_and_CF_Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.ML_recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.ML_recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.ML_recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from Recommenders.ML_recommenders.MatrixFactorization.PureSVD import PureSVDRecommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Hybrids_recommenders.Linear_combination_scores_recommender import Linear_combination_scores_recommender

from Recommenders.ParameterTuning.AbstractClassSearch import DictionaryKeys
from Recommenders.ParameterTuning.BayesianSkoptSearch import BayesianSkoptSearch
from skopt.space import Real, Integer, Categorical


import traceback, pickle
from Recommenders.Utilities.Utils.PoolWithSubprocess import PoolWithSubprocess










def run_KNNCFRecommender_on_similarity_type(similarity_type, parameterSearch,
                                            URM_train, n_cases,
                                            output_folder_path,
                                            output_file_name_root,
                                            metric_to_optimize):



    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
    hyperparamethers_range_dictionary["shrink"] = Integer(0, 1000)
    hyperparamethers_range_dictionary["similarity"] = Categorical([similarity_type])
    hyperparamethers_range_dictionary["normalize"] = Categorical([True, False])

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["tversky_beta"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])


    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases = n_cases,
                                             output_folder_path = output_folder_path,
                                             output_file_name_root = output_file_name_root + "_" + similarity_type,
                                             metric_to_optimize = metric_to_optimize)





def run_KNNCBFRecommender_on_similarity_type(similarity_type, parameterSearch,
                                            URM_train, ICM_train, n_cases,
                                            output_folder_path,
                                            output_file_name_root,
                                            metric_to_optimize):

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
    hyperparamethers_range_dictionary["shrink"] = Integer(0, 1000)
    hyperparamethers_range_dictionary["similarity"] = Categorical([similarity_type])
    hyperparamethers_range_dictionary["normalize"] = Categorical([True, False])

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["tversky_beta"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])

    if similarity_type in ["cosine", "asymmetric"]:
        hyperparamethers_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])



    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [ICM_train, URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases = n_cases,
                                             output_folder_path = output_folder_path,
                                             output_file_name_root = output_file_name_root + "_" + similarity_type,
                                             metric_to_optimize = metric_to_optimize)





def runParameterSearch_Content(recommender_class, URM_train, ICM_object, ICM_name, n_cases = 30,
                             evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                             output_root_path ="result_experiments/", parallelizeKNN = False):


    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)





   ##########################################################################################################

    this_output_root_path = output_root_path + recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    parameterSearch = BayesianSkoptSearch(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

    run_KNNCBFRecommender_on_similarity_type_partial = partial(run_KNNCBFRecommender_on_similarity_type,
                                                   parameterSearch = parameterSearch,
                                                   URM_train = URM_train,
                                                   ICM_train = ICM_object,
                                                   n_cases = n_cases,
                                                   output_root_path = this_output_root_path,
                                                   metric_to_optimize = metric_to_optimize)



    if parallelizeKNN:
        pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        resultList = pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

    else:

        for similarity_type in similarity_type_list:
            run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)








def runParameterSearch_Collaborative(recommender_class, URM_train, metric_to_optimize = "PRECISION",
                                     evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                                     output_folder_path ="result_experiments/", parallelizeKNN = True, n_cases = 30):


    from Recommenders.ParameterTuning.AbstractClassSearch import DictionaryKeys


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME

        parameterSearch = BayesianSkoptSearch(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)




        if recommender_class in [TopPop, Random]:

            recommender = recommender_class(URM_train)

            recommender.fit()

            output_file = open(output_folder_path + output_file_name_root + "_BayesianSearch.txt", "a")
            result_dict, result_baseline = evaluator_validation.evaluateRecommender(recommender)
            output_file.write("ParameterSearch: Best result evaluated on URM_validation. Results: {}".format(result_baseline))

            pickle.dump(result_dict.copy(),
                        open(output_folder_path + output_file_name_root + "_best_result_validation", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            result_dict, result_baseline = evaluator_test.evaluateRecommender(recommender)
            output_file.write("ParameterSearch: Best result evaluated on URM_test. Results: {}".format(result_baseline))

            pickle.dump(result_dict.copy(),
                        open(output_folder_path + output_file_name_root + "_best_result_test", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)


            output_file.close()

            return



        ##########################################################################################################

        if recommender_class is UserKNNCFRecommender:

            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                           parameterSearch = parameterSearch,
                                                           URM_train = URM_train,
                                                           n_cases = n_cases,
                                                           output_folder_path = output_folder_path,
                                                           output_file_name_root = output_file_name_root,
                                                           metric_to_optimize = metric_to_optimize)



            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(2), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)


            return



        ##########################################################################################################

        if recommender_class is ItemKNNCFRecommender:

            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                           parameterSearch = parameterSearch,
                                                           URM_train = URM_train,
                                                           n_cases = n_cases,
                                                           output_folder_path = output_folder_path,
                                                           output_file_name_root = output_file_name_root,
                                                           metric_to_optimize = metric_to_optimize)


            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(2), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)


            return


       ##########################################################################################################

        if recommender_class is P3alphaRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            hyperparamethers_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        ##########################################################################################################

        if recommender_class is RP3betaRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            hyperparamethers_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["beta"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is MatrixFactorization_FunkSVD_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"])
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 150)
            hyperparamethers_range_dictionary["reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":5, "stop_on_validation":True,
                                                                       "evaluator_object":evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed":20, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"])
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 150)
            hyperparamethers_range_dictionary["batch_size"] = Categorical([1])
            hyperparamethers_range_dictionary["positive_reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["negative_reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'positive_threshold':0},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":5, "stop_on_validation":True,
                                                                       "evaluator_object":evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed":20, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is PureSVDRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 250)

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        #########################################################################################################

        if recommender_class is SLIM_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(50, 300)
            hyperparamethers_range_dictionary["epochs"] = Integer(150, 300)
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"])
            hyperparamethers_range_dictionary["lambda_i"] = Real(low = 1e-04, high = 0.01, prior = 'log-uniform')
            hyperparamethers_range_dictionary["lambda_j"] = Real(low = 1e-04, high = 0.01, prior = 'log-uniform')

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights':False, 'symmetric':True, 'positive_threshold':1},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":50, "stop_on_validation":True,
                                                                       "evaluator_object":evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed":10, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is MultiThreadSLIM_ElasticNet:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(50, 200)
            hyperparamethers_range_dictionary["l1_penalty"] = Real(low=1e-06, high=1e-03, prior='log-uniform')
            hyperparamethers_range_dictionary["l2_penalty"] = Real(low=1e-05, high=1e-01, prior='log-uniform')
            hyperparamethers_range_dictionary["alpha"] = Real(low=1e-05, high=1e-01, prior='log-uniform')

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



       #########################################################################################################

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        best_parameters = parameterSearch.search(recommenderDictionary,
                                                 n_cases = n_cases,
                                                 output_folder_path = output_folder_path,
                                                 output_file_name_root = output_file_name_root,
                                                 metric_to_optimize = metric_to_optimize, save_model="no")




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()














import os, multiprocessing
from functools import partial



def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    utility = Data_matrix_utility()
    urm_complete = utility.build_urm_matrix()
    icm_complete = utility.build_icm_matrix()
    urm_temp_train, urm_test = train_test_holdout(URM_all=urm_complete)

    urm_train, urm_validation = train_test_holdout(URM_all=urm_temp_train)
    """
    best = MultiThreadSLIM_ElasticNet(urm_train)
    topK = 150
    l1_value = 1e-05
    l2_value = 0.002
    best.fit(alpha=l1_value + l2_value, l1_penalty=l1_value,
                    l2_penalty=l2_value, topK=topK)
    print("best so far")
    print(best.evaluateRecommendations(URM_test=urm_test, at=10))
    """
    output_folder_path = "result_experiments/"


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)





# best config elasticnet: {'topK': 200, 'l1_penalty': 1.8986406043137196e-06, 'l2_penalty': 0.011673969837199876, 'alpha': 0.0008868749995645901}
#best config SLIM BPR : 'topK': 66, 'epochs': 150, 'sgd_mode': 'adagrad', 'lambda_i': 0.00010838189443121538, 'lambda_j': 0.00011644401162151351
    collaborative_algorithm_list = [
        SLIM_BPR_Cython
    ]




    from Recommenders.Utilities.Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator_validation = SequentialEvaluator(urm_validation, cutoff_list=[5])
    evaluator_test = SequentialEvaluator(urm_test, cutoff_list=[5, 10])


    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = urm_train,
                                                       metric_to_optimize = "MAP",
                                                       n_cases = 15,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_folder_path)





    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)



    for recommender_class in collaborative_algorithm_list:

        try:

            runParameterSearch_Collaborative_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()








if __name__ == '__main__':


    read_data_split_and_search()
