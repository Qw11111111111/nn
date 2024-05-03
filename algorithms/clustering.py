from utils.maths import l2, center_scale
from typing import Literal
import numpy as np
from copy import deepcopy
from utils.utils import argwhere, timeit

class KMeans():
    
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
            indices = np.random.randint(0, X.shape[0], size=self.centroids.shape[0])
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
                for index_, prob in enumerate(cumprobs):
                    if random_number < prob:
                        break
                self.centroids[i] = X[index_]
        
        self.last_centroids = np.zeros_like(self.centroids)

    @timeit
    def fit_predict(self, X: np.ndarray) -> list[int | np.ndarray]:
        # if self.clusters is None: find  the optimal number of clusters. Need to read up on this.
        
        # center and scale the data if needed
        # care: the returned centroid positions will be centered and scaled. If you want to plot these center the data before fitting or set scale to False
        if self.scale:
            X, mean, std = center_scale(X, verbose = True) 
        
        if self.clusters is None:
            scores = np.zeros(shape=self.max_clusters)
            minimum = -np.inf
            for k in range(1, self.max_clusters):
                self.clusters = k
                scores[k] = self._fit(X, get_score = True)
                if scores[k] > minimum:
                    best_centroids, best_assignments = self.centroids, self.centroid_assignment
                    minimum = scores[k]
                if scores[k] >= self.good_score:
                    break
            self.centroids, self.centroid_assignment = best_centroids, best_assignments
        else:
            self._fit(X)
        
        return self.centroid_assignment, self.centroids

    def _update(self, X: np.ndarray) -> None:
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

        # return a list of the clusters for each datpoint (and the positions of the centroids?)
        self.centroids, self.centroid_assignment = best_centroids, best_assignment
        
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
            silhouette_scores[i] = ((b - a) / (np.amax([a, b]))) if np.amax([a, b]) != 0 else 1
        return np.mean(silhouette_scores) if mean else silhouette_scores

