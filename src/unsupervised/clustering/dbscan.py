from src.parents import Clusterer
import numpy as np
from src.utils import timeit
from src.maths import l2

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