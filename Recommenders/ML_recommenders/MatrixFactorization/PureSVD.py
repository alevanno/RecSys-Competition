#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.Base.Recommender_utils import check_matrix
from Recommenders.Utilities.Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.data_matrix import Data_matrix_utility

from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sps


class PureSVDRecommender(Recommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, URM_train):
        super(PureSVDRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')

        self.compute_item_score = self.compute_score_SVD


    def fit(self, num_factors=100):

        from sklearn.utils.extmath import randomized_svd

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition...")

        self.U, self.Sigma, self.VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state=3)

        self.s_Vt = sps.diags(self.Sigma)*self.VT

        self.W_sparse = self.U.dot(self.s_Vt)

        print(self.W_sparse.shape)
        print(type(self.W_sparse))

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition... Done!")

        # truncatedSVD = TruncatedSVD(n_components = num_factors)
        #
        # truncatedSVD.fit(self.URM_train)
        #
        # truncatedSVD

        #U, s, Vt =



    def compute_score_SVD(self, user_id_array):

        try:

            item_weights = self.U[user_id_array, :].dot(self.s_Vt)

        except:
            pass

        return item_weights

    def saveModel(self, folder_path, file_name = None):
        
        import pickle

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = {
            "U":self.U,
            "Sigma":self.Sigma,
            "VT":self.VT,
            "s_Vt":self.s_Vt
        }


        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)


        print("{}: Saving complete")

if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_complete = utility.build_urm_matrix()
    icm_complete = utility.build_icm_matrix().tocsr()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)

    recommender = PureSVDRecommender(URM_train=urm_train)
    i=0
    while i <= 5:
        recommender.fit()
        print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
        i += 1
