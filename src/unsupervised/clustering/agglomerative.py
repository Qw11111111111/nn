from src.parents import Clusterer
import numpy as np
import jax.numpy as jnp
from src.maths import l2, center_scale, mean_of_cluster
from src.utils import timeit

class AgglomerativeClusterer(Clusterer):

    def __init__(self, clusters: int = 2, distance: object = l2, measurement: object = mean_of_cluster) -> None:
        self.prox_matrix = None
        self.distance = distance
        self.measurement = measurement
        self.clusters = []
        self.original = []
        self.matrix = []
        self.n_clusters = clusters
        self.history = []
    
    @timeit
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        data = X
        self.original = X
        self.clusters = [[i] for i in range(data.shape[0])]
        self.prox_matrix = np.zeros(shape=(X.shape[0], X.shape[0]))
        while len(self.clusters) > self.n_clusters:
            minimum = np.inf
            for i, cluster_1 in enumerate(self.clusters):
                for j, cluster_2 in enumerate(self.clusters):
                    if i == j:
                        continue
                    distance = self.distance(self.measurement(data[cluster_1]), self.measurement(data[cluster_2]))
                    if distance < minimum:
                        minimum = distance
                        best = (i, j)
            #self._update_prox_matrix(best)
            self._update_clusters(best)
        return self.clusters

    def _update_clusters(self, pair: tuple) -> None:
        # the cluster will always be shifted to the index of its member with the smallest index and this member will alway be the first -> can be used as reference
        placeholder = [[] for _ in range(len(self.clusters) - 1)]
        for i, cluster_1 in enumerate(self.clusters):
            if i not in pair:
                if i >= pair[1]:
                    placeholder[i - 1] = cluster_1
                else:
                    placeholder[i] = cluster_1
            else:
                if i == pair[0]:
                    placeholder[i] = cluster_1 + self.clusters[pair[1]]
                    next_index = i
        self.clusters = placeholder
        self.history.append(next_index)

    def _update_prox_matrix(self, pair: tuple) -> None:
        cluster_1, cluster_2 = self.clusters[pair[0]], self.clusters[pair[1]]
        for i, point_1 in enumerate(cluster_1):
            for j, point_2 in enumerate(cluster_2):
                self.prox_matrix[point_1][point_2] = np.max((len(cluster_1), len(cluster_2)))
        
    def transform_prox_matrix(self, depth: int) -> list[set[int]]:
        """work in progress"""
        self.clusters_at_depth = []
        prox_matrix = self.prox_matrix
        for i, row in enumerate(prox_matrix):
            for j, number in enumerate(row):
                if number <= depth:
                    self.clusters_at_depth.append(set([j, i]))
                    #self.clusters_at_depth[-1].update(i)
                    prox_matrix[i][j], prox_matrix[j][i] = np.inf, np.inf
                    self._get_numbers(prox_matrix, i, j, depth)
        return self.clusters_at_depth
    
    def _get_numbers(self, prox_matrix: np.ndarray, row: int, col: int, depth: int) -> None:
        for i, num in enumerate(prox_matrix[row]):
            if num <= depth:
                self.clusters_at_depth[-1].update([i])
                prox_matrix[row][i], prox_matrix[i][row] = np.inf, np.inf
                self._get_numbers(prox_matrix, row, i, depth)
        for i, num in enumerate(prox_matrix.T[col]):
            if num <= depth:
                self.clusters_at_depth[-1].update([i])
                prox_matrix[col][i], prox_matrix[i][col] = np.inf, np.inf
                self._get_numbers(prox_matrix.T, i, col, depth)
