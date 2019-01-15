#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.Utilities.Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.CBF_and_CF_Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.CBF_and_CF_Recommenders.Item_based_CF_Recommender import Item_based_CF_recommender
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetMultiProcess import MultiThreadSLIM_ElasticNet
from Recommenders.ML_recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.ML_recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

import numpy as np


class ItemKNNSimilarityHybridRecommender(SimilarityMatrixRecommender, Recommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Similarity_1, Similarity_2, sparse_weights=True):
        super(ItemKNNSimilarityHybridRecommender, self).__init__()

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')

        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.sparse_weights = sparse_weights


    def fit(self, topK=100, alpha = 0.5, beta=0.5):

        self.topK = topK
        self.alpha = alpha
        self.beta = beta

        W = self.Similarity_1*self.alpha + self.Similarity_2*beta

        if self.sparse_weights:
            self.W_sparse = similarityMatrixTopK(W, forceSparseOutput=True, k=self.topK)
        else:
            self.W = similarityMatrixTopK(W, forceSparseOutput=False, k=self.topK)

if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_complete = utility.build_urm_matrix()
    icm_complete = utility.build_icm_matrix().tocsr()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)

    """
    bpr = SLIM_BPR_Cython(urm_train)
    bpr.fit(epochs=250, lambda_i=0.001, lambda_j=0.001, learning_rate=0.01)
    print("BPR performance")
    print(bpr.evaluateRecommendations(URM_test=urm_test, at=10))
    """

    """
    graph = RP3betaRecommender(urm_train)
    graph.fit(topK=100, alpha=0.95, beta=0.3)
    print("Graph performance")
    print(graph.evaluateRecommendations(URM_test=urm_test, at=10))
    """

    """
    elastic_new = MultiThreadSLIM_ElasticNet(urm_train)
    elastic_new.fit(alpha=0.0008868749995645901, l1_penalty=1.8986406043137196e-06,
                    l2_penalty=0.011673969837199876, topK=200)
    print("elastic performance")
    print(elastic_new.evaluateRecommendations(URM_test=urm_test, at=10))
    """

    """
    item_based = Item_based_CF_recommender(urm_train)
    item_based.fit(topK=150, shrink=20)
    print("item based performance")
    print(item_based.evaluateRecommendations(URM_test=urm_test, at=10))
    """

    cbf_new = ItemKNNCBFRecommender(ICM=icm_complete, URM_train=urm_train)
    cbf_new.fit(topK=50, shrink=100, feature_weighting="TF-IDF")
    print("cbf performances")
    print(cbf_new.evaluateRecommendations(URM_test=urm_test, at=10))


    hybrid = ItemKNNSimilarityHybridRecommender(URM_train=urm_train, Similarity_1=elastic_new.W_sparse,
                                                Similarity_2=cbf_new.W_sparse)


    alpha_list = [0.95, 0.96, 0.97, 0.98]
    
    for alpha in alpha_list:
        print("alpha = " + str(alpha) + ", beta = " + str(1.0-alpha))
        hybrid.fit(alpha=alpha, beta=1.0-alpha, topK=250)
        print(hybrid.evaluateRecommendations(URM_test=urm_test, at=10))

    """
    topK_list = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]   
    for k in topK_list:
        print("topk = " + str(k))
        hybrid.fit(alpha=0.55, beta=0.45, topK=k)
        print(hybrid.evaluateRecommendations(URM_test=urm_test, at=10))
    """




"""
ITEM CBF
alpha = 0.8
beta = 0.2
topK = 150
score = 0.1119

ELASTICNET (NEW)
alpha = 0.95
beta = 0.05
topK = 250
score = 0.1200

RP3Beta
alpha = 0.97
beta = 0.03
topK = 200
score = 0.1153

BPR
alpha = 0.55
beta = 0.45
topK = 300
score = 0.1180

"""