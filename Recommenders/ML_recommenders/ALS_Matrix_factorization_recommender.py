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
    reg_list = [0.1, 0.01, 0.001, 0.0001]
    for i in reg_list:
        print("Regularization=" + str(i))
        recommender = IALS_numpy(urm_csr=urm_train, reg=i)
        recommender.fit()
        print(evaluate_algorithm(URM_test=urm_test, recommender_object=recommender, at=10))