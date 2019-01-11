import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from Recommenders.Utilities.Base.Recommender import Recommender
from Recommenders.Utilities.Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.Utilities.data_matrix import Data_matrix_utility
from Recommenders.Utilities.data_splitter import train_test_holdout

from Recommenders.Utilities.Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import time, sys




class Content_P3alphaRecommender(SimilarityMatrixRecommender, Recommender):
    """ P3alpha recommender """

    RECOMMENDER_NAME = "Content_P3alphaRecommender"

    """
    VERSIONS:
    - 'user' -> Final matrix is computed as Pui * Piu * Pui
    - 'content' -> Final matrix is computed as Pui * Pia * Pai
    - 'content-user' -> Final matrix is computed as Pui * Pia * Pai * Piu * Pui
    - 'user-content' -> Final matrix is computed as Pui * Piu * Pui * Pia * Pai
    - 'content-user-content' -> Final matrix is computed as Pui * Pia * Pai * Piu * Pui * Pia * Pai
    
    
    
    """

    def __init__(self, URM_train, ICM, version='user'):
        super(Content_P3alphaRecommender, self).__init__()

        print("Version: " + version)
        self.URM_train = check_matrix(URM_train, format='csr', dtype=np.float32)
        self.ICM = ICM
        self.sparse_weights = True
        self.version = version
        if self.version != 'user' and self.version != 'content' and self.version != 'content-user' \
            and self.version != 'user-content' and self.version != 'content-user-content':
            print("ERROR")


    def __str__(self):
        return "Content_P3alpha(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                            self.min_rating, self.topK, self.implicit,
                                                                            self.normalize_similarity)

    def fit(self, topK=100, alpha=1., min_rating=0, implicit=False, normalize_similarity=False):

        self.topK = topK
        self.alpha = alpha
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity


        #
        # if X.dtype != np.float32:
        #     print("P3ALPHA fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        #Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32) #cast from int to float32
        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        Pia = normalize(self.ICM, norm='l1', axis=1)
        X_bool = self.ICM.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        Pai = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)
            Pia = Pia.power(self.alpha)
            Pai = Pai.power(self.alpha)

        # Final matrix is computed as Pui * Pia * Pai
        # Multiplication unpacked for memory usage reasons
        block_dim = 200

        if self.version == 'user' or self.version == 'user-content':
            d_t = Piu
        else:
            d_t = Pia

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0


        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            if self.version == 'user':
                similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            elif self.version == 'content':
                similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pai
            elif self.version == 'content-user':
                similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pai * Piu * Pui
            elif self.version == 'user-content':
                similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui * Pia * Pai
            else:
                similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pai * Piu * Pui * Pia * Pai


            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1


            if time.time() - start_time_printBatch > 60:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))


        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)


        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, forceSparseOutput = True, k=self.topK)
            self.sparse_weights = True

if __name__ == '__main__':
    utility = Data_matrix_utility()
    #provide_recommendations(utility.build_urm_matrix())

    urm_complete = utility.build_urm_matrix()
    icm = utility.build_icm_matrix().tocsr()
    urm_train, urm_test = train_test_holdout(URM_all=urm_complete)
    content_graph_recommender = Content_P3alphaRecommender(URM_train=urm_train, ICM=icm, version='user')
    content_graph_recommender.fit(topK=250, alpha=0.95)
    print(content_graph_recommender.evaluateRecommendations(URM_test=urm_test))

    content_graph_recommender = Content_P3alphaRecommender(URM_train=urm_train, ICM=icm, version='content-user')
    topk_list = [250]
    for topk in topk_list:
        print("topk=" + str(topk))
        content_graph_recommender.fit(topK=topk, alpha=0.8)
        print(content_graph_recommender.evaluateRecommendations(URM_test=urm_test))
