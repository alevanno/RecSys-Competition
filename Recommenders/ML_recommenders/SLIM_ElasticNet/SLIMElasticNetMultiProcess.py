import multiprocessing
from multiprocessing import Pool
from functools import partial
import threading
from Recommenders.ML_recommenders.SLIM_ElasticNet.SLIMElasticNetRecommender import  SLIMElasticNetRecommender
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.data_splitter import train_test_holdout
import numpy as np
import scipy.sparse as sps

from Recommenders.Utilities.Base.Recommender_utils import check_matrix
from Recommenders.Utilities.Base.SimilarityMatrixRecommender import *
from sklearn.linear_model import ElasticNet

class MultiThreadSLIM_ElasticNet(SLIMElasticNetRecommender, SimilarityMatrixRecommender):

    def __init__(self, URM_train):
        super(MultiThreadSLIM_ElasticNet, self).__init__(URM_train)

    def __str__(self):
        return "SLIM_mt (l1_penalty={},l2_penalty={},positive_only={},workers={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only, self.workers
        )

    def _partial_fit(self, currentItem, X, topK, alpha=1.0):
        model = ElasticNet(alpha=alpha,
                           l1_ratio=self.l1_ratio,
                           positive=self.positive_only,
                           fit_intercept=False,
                           copy_X=False,
                           precompute=True,
                           selection='random',
                           max_iter=100,
                           tol=1e-4)

        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, currentItem].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0
        # fit one ElasticNet model per column
        model.fit(X_j, y)
        # self.model.coef_ contains the coefficient of the ElasticNet model
        # let's keep only the non-zero values
        # nnz_idx = model.coef_ > 0.0

        relevant_items_partition = (-model.coef_).argpartition(topK)[0:topK]
        relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        notZerosMask = model.coef_[ranking] > 0.0
        ranking = ranking[notZerosMask]

        values = model.coef_[ranking]
        rows = ranking
        cols = [currentItem] * len(ranking)

        #
        # values = model.coef_[nnz_idx]
        # rows = np.arange(X.shape[1])[nnz_idx]
        # cols = np.ones(nnz_idx.sum()) * currentItem
        #
        return values, rows, cols

    def fit(self, alpha=1.0, l1_penalty=0.1,
            l2_penalty=0.1,
            positive_only=True,
            topK=100,
            workers=multiprocessing.cpu_count()):
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        self.topK = topK

        self.workers = workers

        try:

            self.W_sparse = self.loadModel()
            self.W_sparse = self.W_sparse.tocsr()

        except IOError:

            URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
            n_items = self.URM_train.shape[1]
            # fit item's factors in parallel

            # oggetto riferito alla funzione nel quale predefinisco parte dell'input
            _pfit = partial(self._partial_fit, X=URM_train, topK=self.topK, alpha=alpha)

            # creo un pool con un certo numero di processi
            pool = Pool(processes=self.workers)

            # avvio il pool passando la funzione (con la parte fissa dell'input)
            # e il rimanente parametro, variabile
            res = pool.map(_pfit, np.arange(n_items))

            # res contains a vector of (values, rows, cols) tuples
            values, rows, cols = [], [], []
            for values_, rows_, cols_ in res:
                values.extend(values_)
                rows.extend(rows_)
                cols.extend(cols_)

            # generate the sparse weight matrix
            self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
            self.saveModel()

    def saveModel(self, file_name='SLIM_ElasticNet.npz'):
        sps.save_npz(file=file_name, matrix=self.W_sparse)

    def loadModel(self, file_name='SLIM_ElasticNet.npz'):
        return sps.load_npz(file=file_name)

############################################################################################################

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_array = utility.get_target_list()
    l1_value = 1e-05
    l2_value = 0.002
    recommender = MultiThreadSLIM_ElasticNet(urm_csr)
    recommender.fit(alpha=l1_value+l2_value, l1_penalty=l1_value, \
            l2_penalty=l2_value)
    recommended_list = recommender.recommend(user_id_array=targets_array, cutoff=10)
    for index in range(len(targets_array)):
        recommendations[targets_array[index]] = recommended_list[index]

    with open('slimElasticNet_recommendations.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommendations):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommendations[i]])))


if __name__ == '__main__':
    utility = Data_matrix_utility()
    #provide_recommendations(utility.build_matrix())

    urm_complete = utility.build_urm_matrix()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    print("Train: " + str(urm_train.shape))
    print("Test: " + str(urm_test.shape))
    topK_array = [150]
    l1_value = 1e-05
    l2_value = 0.002
    recommender = MultiThreadSLIM_ElasticNet(urm_train)
    print(recommender.URM_train.getformat())
    for k in topK_array:
        print("k=" + str(k))
        recommender.fit(alpha=l1_value+l2_value, l1_penalty=l1_value,\
                l2_penalty=l2_value, topK=k)

        for id in range(urm_train.shape[0]):
            scores = recommender.compute_item_score(id)
            print("Minimum: " + str(scores[np.nonzero(scores)].min()))
            print("Maximum: " + str(scores.max()))
            print("mean: " + str(scores[np.nonzero(scores)].mean()))

        print(recommender.evaluateRecommendations(URM_test=urm_test, at=10))
        print("Ignoring users with few interactions...")
        print(recommender.evaluateRecommendations(URM_test=urm_test, at=10, filterCustomUsers=utility.user_to_neglect()))
        print("Ignoring users with large interaction")
        print(recommender.evaluateRecommendations(URM_test=urm_test, at=10, filterCustomUsers=utility.user_to_neglect(intensity_interaction='large')))