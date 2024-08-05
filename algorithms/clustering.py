from parents.parentclasses import Clusterer
from utils.maths import l2, center_scale, mean_of_cluster
from typing import Literal
import numpy as np
from copy import deepcopy
from utils.utils import argwhere, timeit, unique_nums

class KMeans(Clusterer):
    
    def __init__(self, num_clusters: int | None = None, n_retries: int = 10, verbose: bool = False, scale: bool = True, init_method: Literal["kmeans++", "random", "random_choice"] = "kmeans++", max_clusters: int = 20, good_score: float = 0.7) -> None:
        self.clusters = num_clusters
        self.centroid_assignment = None
        self.centroids = None
        self.retries = n_retries
        self.verbose = verbose
        self.scale = scale
        self.init_method = init_method
        self.max_clusters = max_clusters
        self.good_score = good_score

    def _initialize_centroids(self, X: np.ndarray) -> None:
        self.centroids = np.zeros(shape=(self.clusters, X.shape[1]))
        if self.init_method == "random":
            for i in range(self.clusters):
                self.centroids[i] += np.random.normal(scale = np.std(X, axis=0) ** 2, loc = np.mean(X, axis=0))

        elif self.init_method == "random_choice":
            while not unique_nums(indices := np.random.randint(0, X.shape[0], size=self.centroids.shape[0])):
                pass
            self.centroids = X[indices]
        
        elif self.init_method == "kmeans++":
            self.centroids[0] = X[np.random.randint(X.shape[0])]
            for i in range(1, self.centroids.shape[0]):
                probs = np.zeros(shape = X.shape[0])
                for j, point in enumerate(X):
                    if np.sum([self.centroids[i] == point for i in range(self.centroids.shape[0])]):
                        continue
                    closest_centroid = self.centroids[np.argmin([l2(point, self.centroids[k]) if not self.centroids[k].all() == 0 else np.inf for k in range(self.centroids.shape[0])])]
                    probs[j] = l2(point, closest_centroid)
                probs /= np.sum(probs)
                cumprobs = probs.cumsum()
                random_number = np.random.random()
                for index, prob in enumerate(cumprobs):
                    if random_number < prob:
                        break
                self.centroids[i] = X[index]
        
        self.last_centroids = np.zeros_like(self.centroids)
        self.initial_centroids = deepcopy(self.centroids)

    @timeit
    def fit_predict(self, X: np.ndarray) -> list[int | np.ndarray]:
        # center and scale the data if needed
        # care: the returned centroid positions will be centered and scaled. If you want to plot these center the data before fitting or set scale to False
        if self.scale:
            X, mean, std = center_scale(X, verbose = True) 
        
        if self.clusters is None:
            scores = np.zeros(shape=self.max_clusters + 1)
            minimum = -np.inf
            for k in range(1, self.max_clusters + 1):
                self.clusters = k
                scores[k] = self._fit(X, get_score = True)
                if scores[k] > minimum:
                    best_centroids, best_assignments, best_initial_centroids = deepcopy(self.centroids), deepcopy(self.centroid_assignment), deepcopy(self.initial_centroids)
                    minimum = scores[k]
                if scores[k] >= self.good_score:
                    break
            self.centroids, self.centroid_assignment, self.best_initial_centroids = best_centroids, best_assignments, best_initial_centroids
        else:
            self._fit(X)
        self._update_partitions(X)
        return self.centroid_assignment, self.centroids

    def _update_centroids(self, X: np.ndarray) -> None:
        self.last_centroids = deepcopy(self.centroids)
        for i, centroid in enumerate(self.centroids):
            datapoints = self.centroid_assignment[i]
            if len(datapoints) == 0:
                continue
            # calculating the new coordinates via the mean of the associated points
            #try:
            self.centroids[i] = np.hstack([np.mean(X[datapoints][:,coord], axis=0) for coord in range(X.shape[1])])
            #except IndexError:
                #continue

    def _update_partitions(self, X: np.ndarray) -> None:
        # assign  all points to the cluster with the smallest distance to its centroid and repeat until no more changes can be made.
        self.centroid_assignment = [[] for _ in range(self.clusters)]
        for i, point in enumerate(X):
            best = np.argmin([l2(point, centroid) for j, centroid in enumerate(self.centroids)])
            self.centroid_assignment[best].append(i)

    def _update(self, X: np.ndarray) -> None:
            
            self._update_partitions(X)
            self._update_centroids(X)
            return

            # assign  all points to the cluster with the smallest distance to its centroid and repeat until no more changes can be made.
            self.centroid_assignment = [[] for _ in range(self.clusters)]
            for i, point in enumerate(X):
                best = np.argmin([l2(point, centroid) for j, centroid in enumerate(self.centroids)])
                self.centroid_assignment[best].append(i)
            
            # update the centroids using the average of all points assigned to it as a new centroid
            self.last_centroids = self.centroids
            for i, centroid in enumerate(self.centroids):
                datapoints = self.centroid_assignment[i]
                if len(datapoints) == 0:
                    continue
                # calculating the new coordinates via the mean of the associated points
                try:
                    self.centroids[i] = np.hstack([np.mean(X[datapoints][:,coord], axis=0) for coord in range(X.shape[1])])
                except IndexError:
                    continue
    
    def _fit(self, X: np.ndarray, get_score: bool = False) -> float | None:
        # initialize centroids randomly
        # list of datapoints
        minimum = np.inf
        for trial in range(self.retries):
            self.centroid_assignment = [[] for _ in range(self.clusters)]

            #positions of centroids
            self._initialize_centroids(X)

            counter = 0
            while counter <= 2:
                while (self.centroids - self.last_centroids).any() != 0:
                    self._update(X)
                else:
                    counter += 1
        
            total = 0
            for i, cluster in enumerate(self.centroid_assignment):
                if len(cluster) < 1:
                    continue
                total += np.sum([l2(self.centroids[i], X[point]) for point in cluster])
            if self.verbose:
                print(f"Epoch: {trial} | current: {total:.3f} | min: {minimum:.3f}")
            if total < minimum:
                minimum = total
                best_assignment = deepcopy(self.centroid_assignment)
                best_centroids = deepcopy(self.centroids)
                best_initial_centroids = deepcopy(self.initial_centroids)

        # return a list of the clusters for each datpoint (and the positions of the centroids?)
        self.centroids, self.centroid_assignment, self.initial_centroids = best_centroids, best_assignment, best_initial_centroids
        
        if get_score:
            score = self.appr_silhouette(X)
            return score

    #@timeit
    def silhouette(self, X: np.ndarray, mean: bool = True) -> float | np.ndarray:
        """Calculate the Silhouette Coefficient for each sample. or the mean of it. USE APPR_SILHOUETTE INSTEAD"""
        # does not work. Use approximation instead
        #https://en.wikipedia.org/wiki/Silhouette_(clustering)
        silhouette_scores = np.zeros(shape=X.shape[0])
        for i, point in enumerate(X):
            a = 0
            minimum = np.inf
            assignment = argwhere(self.centroid_assignment, i, axis=1)[0]
            for j, cluster in enumerate(self.centroid_assignment):
                if len(self.centroid_assignment[assignment]) <= 1:
                    continue
                b = 0
                if j == assignment:
                    a += np.mean([l2(point, X[datapoint]) if datapoint != i else 0 for datapoint in cluster])
                b += np.mean([l2(point, X[datapoint]) for datapoint in cluster])
                if b < minimum:
                    minimum = b
            silhouette_scores[i] = (minimum - a) / (np.amax([a, minimum])) if len(self.centroid_assignment[assignment]) > 1 else 0
        return np.mean(silhouette_scores) if mean else silhouette_scores
    
    #@timeit
    def appr_silhouette(self, X: np.ndarray, mean: bool = True) -> float | np.ndarray:
        """Apply the Silhouette method on an unsupervised learning model."""
        silhouette_scores = np.zeros(shape=X.shape[0])
        for i, point in enumerate(X):
            minimum = np.inf
            assignment = argwhere(self.centroid_assignment, i, axis=1)[0]
            a = l2(point, self.centroids[assignment])
            b = np.min([l2(point, centroid) if j != assignment else np.inf for j, centroid in enumerate(self.centroids)])
            try:
                silhouette_scores[i] = ((b - a) / (np.amax([a, b]))) if np.amax([a, b]) != 0 else 1 if np.amax([a, b]) != np.inf else 0 if np.amax([a, b]) != np.nan else 0.5 #this still raises and shows a runtime warning. Need to fix
            except RuntimeWarning:
                silhouette_scores[i] = 1 if np.amax([a, b]) == 0 else 0
        return np.mean(silhouette_scores) if mean else silhouette_scores

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

class DBScan(Clusterer):
    #https://de.wikipedia.org/wiki/DBSCAN

    def __init__(self, min_pts: int = 3, epsilon: float = 3e-1) -> None:
        super().__init__()
        self.is_visited = set()
        self.is_noise = set()
        self.cluster_assignments = []
        self.min_pts = min_pts
        self.epsilon = epsilon

    @timeit
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        for i, point in enumerate(X):
            if i in self.is_visited:
                continue
            self.is_visited.add(i)
            neighbours = self._neighbours(X, point)
            if len(neighbours) < self.min_pts:
                self.is_noise.add(i)
            else:
                self.cluster_assignments.append(set())
                self._expand_cluster(X, neighbours, i)

        clusters = []
        for i, point in enumerate(X):
            count = 0
            for j, cluster in enumerate(self.cluster_assignments):
                if i in cluster:
                    clusters.append(j)
                elif i in self.is_noise:
                    count += 1
            if count == len(self.cluster_assignments):
                clusters.append(-1)

        return clusters
    
    def _expand_cluster(self, X: np.ndarray, neighbours: list[int], index: int) -> None:
        self.cluster_assignments[-1].add(index)
        i = 0
        while len(set(neighbours) - self.is_visited) > 0:
            point = neighbours[i]
            if point in self.is_visited:
                i += 1 
                continue
            self.is_visited.add(point)
            neighbours_ = self._neighbours(X, X[point])
            if len(neighbours_) >= self.min_pts:
                neighbours += neighbours_
            self.cluster_assignments[-1].add(point)
            if point in self.is_noise:
                self.is_noise.remove(point)
            
            i += 1

    def _neighbours(self, X: np.ndarray, point: np.ndarray) -> list[int]:
        distances = np.array([l2(point, point_2) for point_2 in X])
        neighbours = distances <= self.epsilon
        indices = np.arange(len(distances))[neighbours]
        return list(indices)
