import numpy as np
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import coo_matrix
from Compute_Similarity_Python import Compute_Similarity_Python
from evaluation_function import evaluate_algorithm
from data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot

train_path = Path("data")/"train.csv"
target_path = Path('data')/'target_playlists.csv'

"""
Here a User-based CF recommender is implemented
"""
################################################################################
class Data_matrix_utility(object):
    def __init__(self, path):
        self.train_path = path

    def build_matrix(self):   #for now it works only for URM
        data = pd.read_csv(self.train_path)
        n_playlists = data.nunique().get('playlist_id')
        n_tracks = data.nunique().get('track_id')

        playlists_array = self.extract_array_from_dataFrame(data, ['track_id'])
        track_array = self.extract_array_from_dataFrame(data, ['playlist_id'])
        implicit_rating = np.ones_like(np.arange(len(track_array)))
        urm = coo_matrix((implicit_rating, (playlists_array, track_array)), \
                            shape=(n_playlists, n_tracks))
        return urm

    def extract_array_from_dataFrame(self, data, columns_list_to_drop):
        array = data.drop(columns=columns_list_to_drop).get_values()
        return array.T.squeeze() #transform a nested array in array and transpose it
################################################################################

################################################################################
class CF_recommender(object):
    def __init__(self, urm_csr):
        self.urm_csr = urm_csr

    def fit(self, topK=50, shrink=100, normalize = True, similarity = "cosine"):
        similarity_object = Compute_Similarity_Python(self.urm_csr.transpose(), shrink=shrink,\
                                                  topK=topK, normalize=normalize,\
                                                  similarity = similarity)
#I pass the transpose of urm to calculate the similarity between playlists.
#I obtain a similarity matrix of dimension = number of playlists * number_of_playlists
        self.similarity_csr = similarity_object.compute_similarity()

    def recommend(self, target_id, n_tracks=None, exclude_seen=True):
        target_profile = self.similarity_csr.getrow(target_id)
        scores = target_profile.dot(self.urm_csr).toarray().ravel()
        if exclude_seen:
            scores = self.filter_seen(target_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:n_tracks]

    def filter_seen(self, target_id, scores):
        start_pos = self.urm_csr.indptr[target_id] #extracts the column in which the target start
        end_pos = self.urm_csr.indptr[target_id+1] #extracts the column in which the target ends
        target_profile = self.urm_csr.indices[start_pos:end_pos] #extracts the columns indexes
                                #in which we can find the non zero values in target
        scores[target_profile] = -np.inf
        return scores
################################################################################
"""
def provide_recommendations(k, urm):
    recommendations = {}
    urm_csr = urm.tocsr()
    targets_df = pd.read_csv(target_path)
    targets_array = targets_df.get_values().squeeze()
    recommender = CF_recommender(urm_csr)
    recommender.fit(shrink=0, topK = k)
    for target in targets_array:
        recommendations[target] = recommender.recommend(target_id=target,n_tracks=10)
"""

if __name__ == '__main__':
    utility = Data_matrix_utility(train_path)
    urm_complete = utility.build_matrix()
    urm_train, urm_test = train_test_holdout(URM_all = urm_complete)
    recommender = CF_recommender(urm_train)

    K_values = [x for x in range(160,195,5)]
    K_results = []
    for k in K_values:
        recommender.fit(topK=k,shrink=0)
        evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
        K_results.append(evaluation_metrics["MAP"])
    """
    shrink_value = [for x in range(10)]
    shrink_result = []
    for value in shrink_value:
        recommender.fit(topK = ,shrink=value)
        evaluation_metrics = evaluate_algorithm(URM_test=urm_test, recommender_object=\
                                         recommender)
        shrink_result.append(evaluation_metrics["MAP"])
    """
    pyplot.plot(K_values, K_results)
    pyplot.ylabel('MAP')
    pyplot.xlabel('TopK')
    pyplot.show()


###############################################################################
"""
Results:
- Execution time = 1.90 min (1.40 to compute similarity + 0.5 for the rest)
- MAP on Public Test Set: 0.07657

TODO:
1. testare su un validation set differenti valori di k (del topk), plottando su un
  grafico come varia il MAP al variare di k e usare il k che massimizzi il MAP
2. Fare lo stesso procedimento del punto 1 per lo shrink (quindi plottare il grafico
del MAP in funzione dello shrink e prendere lo shrink per cui il MAP del validation set
è massimo)
3. Provare a scrivere il CF item based recommender e paragonare le performance rispetto
allo user-based

REFERENCES
Per i primi 2 punti guardare il notebook "Practice 3 - Content Based Filtering"
per avere un'idea su come plottare i grafici. Per dividere il train set in train e
validation sets importare data_splitter.py contenuto in questa stessa cartella.
Per farsi un'idea di quali valori di k e shrink provare nei punti 1 e 2 attenersi
a quanto detto dal prof in classe.
"""
