import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix
from Utilities.evaluation_function import evaluate_algorithm
from Utilities.data_splitter import train_test_holdout
from Utilities.data_matrix import Data_matrix_utility
import matplotlib.pyplot as pyplot

train_path = Path("data")/"train.csv"
target_path = Path('data')/'target_playlists.csv'

if __name__ == '__main__':
    utility = Data_matrix_utility(tracks_path, train_path)
#    icm = utility.build_icm_matrix()
#    urm = utility.build_urm_matrix()
#    provide_recommendations(urm, icm)

    icm = utility.build_icm_matrix()
    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all = urm_complete)
    recommender = Hybrid_recommender(urm_train, icm.tocsr())

    recommender.fit()
    evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
    print(evaluation_metrics)