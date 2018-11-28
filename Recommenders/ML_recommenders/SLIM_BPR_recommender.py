import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix
from Recommenders.Utilities.evaluation_function import evaluate_algorithm
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.ML_recommenders.SLIM_BPR import SLIM_BPR as s
import matplotlib.pyplot as pyplot

if __name__ == '__main__':
    utility = Data_matrix_utility()
    urm_complete = utility.build_urm_matrix()

    urm_train, urm_test = train_test_holdout(URM_all = urm_complete)
    recommender = s.SLIM_BPR(URM_train = urm_train)

    recommender.fit()
    print(recommender.evaluateRecommendations(URM_test = urm_test))