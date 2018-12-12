import sys
import numpy as np
import scipy.sparse as sps

from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.Base.Recommender_utils import check_matrix
from Recommenders.Utilities.Base.SimilarityMatrixRecommender import *
from sklearn.linear_model import ElasticNet
from Recommenders.Utilities.data_splitter import train_test_holdout
import time
from Recommenders.Utilities.data_matrix import Data_matrix_utility


################################################################################
class SLIMElasticNetRecommender(SimilarityMatrixRecommender, Recommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available
    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides
        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

    def __init__(self, URM_train):
        super(SLIMElasticNetRecommender, self).__init__()
        self.URM_train = URM_train
        self.debug_weights = {}

    def __str__(self):
        return "SLIM (l1_penalty={},l2_penalty={},positive_only={})".format(\
            self.l1_penalty, self.l2_penalty, self.positive_only)

    def fit(self, alpha=1.0,l1_penalty=0.1, l2_penalty=0.1, positive_only=True, topK = 100):

        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.topK = topK

        if self.l1_penalty + self.l2_penalty != 0:
            self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        else:
            print("SLIM_ElasticNet: l1_penalty+l2_penalty cannot be equal to zero, setting the ratio l1/(l1+l2) to 1.0")
            self.l1_ratio = 1.0

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        print("l1_coefficient= " + str(self.model.alpha*self.l1_ratio))
        print("l2_coefficient= " + str(0.5*self.model.alpha*(1-self.l1_ratio)))

        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = URM_train.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            # nonzero_model_coef_index = self.model.coef_.nonzero()[0]
            # nonzero_model_coef_value = self.model.coef_[nonzero_model_coef_index]

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            self.debug(self.model.sparse_coef_, currentItem)

            local_topK = min(len(nonzero_model_coef_value)-1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]
                numCells += 1

            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            if time.time() - start_time_printBatch > 300 or currentItem == n_items-1:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}".format(\
                                  currentItem+1,\
                                  100.0* float(currentItem+1)/n_items,\
                                  (time.time()-start_time)/60,\
                                  float(currentItem)/(time.time()-start_time)))
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),\
            shape=(n_items, n_items), dtype=np.float32)

        if len(self.debug_weights) > 0:
            csv_name = 'weights_debug' + 'l1=' + str(self.model.alpha*self.l1_ratio)\
                + 'l2=' + str(0.5*self.model.alpha*(1-self.l1_ratio)) + '.csv'
            with open(csv_name, 'w') as d:
                d.write('itemid,mean,dev_std,min,max,sparse\n')
                for i in self.debug_weights:
                    d.write('{},{}\n'.format(i, ' '.join([str(x) for x in self.debug_weights[i]])))

            self.debug_weights.clear()

    def debug(self, coeff_matrix, itemid):
        array = coeff_matrix.toarray().ravel()
        non_zero_array = array[array.nonzero()]
        if non_zero_array.shape[0] == 0:
            self.debug_weights[itemid] = [str(0)+',', str(0)+',',str(0)+',', str(0)\
                +',',str(1)]
        else:
            mean = non_zero_array.mean()
            dev_std = non_zero_array.std()
            min = non_zero_array.min()
            max = non_zero_array.max()
            sparse = 1 - (non_zero_array.shape[0]/array.shape[0])
            self.debug_weights[itemid] = [str(mean)+',', str(dev_std)+',', str(min)+',',\
             str(max)+',',str(sparse)]



################################################################################

def provide_recommendations(urm):
    recommendations = {}
    urm_csr = urm.tocsc()
    targets_array = utility.get_target_list()
    recommender = SLIMElasticNetRecommender(urm_csr)
    l1_value = 1e-05
    l2_value = 0.002
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
    urm_train, urm_test = train_test_holdout(URM_all = urm_complete)
    print("Train: " + str(urm_train.shape))
    print("Test: " + str(urm_test.shape))
    topK_array = [100, 120, 150, 200]
    l1_value = 1e-05
    l2_value = 0.002
    urm_csr = urm_train.tocsr()
    recommender = SLIMElasticNetRecommender(urm_train)
    for k in topK_array:
        print("k=" + str(k))
        recommender.fit(alpha=l1_value+l2_value, l1_penalty=l1_value,\
                l2_penalty=l2_value, topK=k)
        print(recommender.evaluateRecommendations(URM_test=urm_test))


###############################################################################
