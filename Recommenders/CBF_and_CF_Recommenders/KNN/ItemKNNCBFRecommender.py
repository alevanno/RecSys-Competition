#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.Base.Recommender_utils import check_matrix
from Recommenders.Utilities.Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Recommenders.Utilities.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.data_matrix import Data_matrix_utility

import numpy as np

from Recommenders.Utilities.Base.Similarity.Compute_Similarity import Compute_Similarity


class ItemKNNCBFRecommender(SimilarityMatrixRecommender, Recommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCBFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, ICM, URM_train, sparse_weights=True):
        super(ItemKNNCBFRecommender, self).__init__()

        self.ICM = ICM.copy()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.sparse_weights = sparse_weights


    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = okapi_BM_25(self.ICM)

        elif feature_weighting == "TF-IDF":
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = TF_IDF(self.ICM)


        similarity = Compute_Similarity(self.ICM.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)


        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()

if __name__ == '__main__':
    utility = Data_matrix_utility()

    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)

    icm = utility.build_icm_matrix().tocsr()

    recommender = ItemKNNCBFRecommender(ICM=icm, URM_train=urm_train)
    recommender.fit(topK=50, shrink=100, feature_weighting="TF-IDF")
    print("Best so far")
    print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
    """
    topk_list = [50, 75, 100]
    shrink_list = [3, 10, 50, 100, 150]

    for k in topk_list:
        for shrink in shrink_list:
            print("topk=" + str(k) + ", shrink=" + str(shrink))
            recommender.fit(topK=k, shrink=shrink, feature_weighting="BM25")
            print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
    """