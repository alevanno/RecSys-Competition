from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.data_splitter import train_test_holdout
from Recommenders.ML_recommenders.MatrixFactorization.MatrixFactorization_RMSE import IALS_numpy
from Recommenders.Utilities.evaluation_function import evaluate_algorithm

if __name__ == '__main__':
    utility = Data_matrix_utility()
    #provide_recommendations(utility.build_urm_matrix())


    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    nfactor_list = [10, 25, 50, 100, 150, 200, 250, 300]
    for i in nfactor_list:
        print("Factor=" + str(i))
        recommender = IALS_numpy(urm_csr=urm_train, num_factors=i, iters=25)
        recommender.fit()
        print(evaluate_algorithm(URM_test=urm_test, recommender_object=recommender, at=10))