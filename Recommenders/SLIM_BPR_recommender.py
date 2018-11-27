import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix
from Utilities.evaluation_function import evaluate_algorithm
from Utilities.data_splitter import train_test_holdout
from Utilities.data_matrix import Data_matrix_utility
from ML_recommenders.SLIM_BPR import SLIM_BPR as s
import matplotlib.pyplot as pyplot

train_path = Path("data")/"train.csv"
target_path = Path('data')/'target_playlists.csv'

if __name__ == '__main__':
    utility = Data_matrix_utility(train_path)
    urm_complete = utility.build_matrix()

    urm_train, urm_test = train_test_holdout(URM_all = urm_complete)
    recommender = s.SLIM_BPR(URM_train = urm_train)

    recommender.fit()
    print(recommender.evaluateRecommendations(URM_test = urm_test))