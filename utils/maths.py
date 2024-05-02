import numpy as np
from utils.utils import argwhere

def MSELoss(X: np.ndarray | int = None, Y: np.ndarray | int = None, w: np.ndarray | int = None, pred: np.ndarray | int = None, mode: str = "forward") -> float | np.ndarray:
    if X and w:
        pred = np.dot(X, w)
    if mode == "forward":
        return np.sum([(pred[i] - Y[i]) ** 2 for i in range(len(Y))]) / len(Y)
    else:
        return 2 * (pred[0] - Y[0])
    
def loss(X, w, Y, lambda_value: float = 0, func = lambda x: x) ->  float:
    """MSE Loss function, takes a np.array or float PRED and a np.array or float Y and returns the gradient of the MSE loss"""
    if np.isscalar(X):
        X = np.array([X])
    if np.isscalar(Y):
        Y = np.array([Y])
    return - np.dot(np.transpose(X), (Y - np.dot(X, func(w)))) + 2 * lambda_value * func(w)

def l2(w: np.ndarray, y: np.ndarray = None, grad: bool = False, squared: bool = False) -> float:
    #TODO implement correctly
    if grad:
        if squared:
            return 2 * np.sum(w)
        else:
            #TODO implement
            pass
    if squared:
        if y is None:
            return np.sum(np.square(w))
        else:
            return np.sum(np.square(w - y))
    else:
        if y is None:
            return np.sqrt(np.sum(np.square(w)))
        else:
            return np.sqrt(np.sum(np.square(w - y)))
    
def l1(w: np.ndarray, y: np.ndarray = None, grad: bool = False) -> float:
    #TODO check if correct
    if grad:
        return np.sum(np.ones_like(w))
    if not y:
        return np.sum(w)
    else:
        return np.sum(np.square(w - y))

def diagonal(X: np.ndarray, is_diagonalizable: bool = False, svd: bool = True) -> np.ndarray:
    """diagonalizes the input matrix X and returns a diagonal matrix of shape(min(X.shape), min(X.shape))"""
    #TODO: implement diagonalization for square matrices and non square matrices. Check if matrix is diagonalizable.
    # check if matrix is diagonalizable:
    if not is_diagonalizable:
        # not implemented yet
        pass

    if X.shape[0] == X.shape[1]:
        """not implemented yet"""
        pass
    if svd:
        return np.diag(np.linalg.svd(X)[1])
    else:
        return np.diag(np.diag(X))
    
def center_scale(X: np.ndarray, axis: int = 0) -> np.ndarray:
    mean = np.mean(X, axis = axis)
    std = np.std(X, axis=axis)
    return (X - mean) / std
    
def PCA(X: np.ndarray, n_components: int | None = None, axis: int = 0) -> np.ndarray:
    # center and scale the data 
    X = center_scale(X, axis)
    U, S, V = np.linalg.svd(X)
    S = np.diag(S)
    # truncate the result
    if n_components is None:
        # find optimal split https://stackoverflow.com/questions/4471993/compute-the-elbow-for-a-curve-automatically-and-mathematically
        # i belive that this does not currently work correctly
        pivot = np.argmax([S[i + 1] + S[i - 1] - 2 * S[i] for i in range(1, len(S) - 1)])
    else:
        pivot = n_components

    #default to 1 if auto calculates something false
    if pivot < 1:
        pivot = 1

    # generate the truncated SVD
    S = S[:pivot][:pivot]
    U = U[:, :pivot]    
    V = V[:pivot]

    Y = np.zeros(shape=(X.shape[0], S.shape[0]))
    # calculate the new columns of Y with the SVD results according to Eckart-Young’s theorem.
    for i in range(pivot):
        beta = np.dot(S, V[:][i])
        Y[:,i] = np.dot(U, beta) # + mean
    return Y

class KMeans():
    
    def __init__(self, num_clusters: int | None = None, n_retries: int = 1) -> None:
        self.clusters = num_clusters
        self.centroid_assignment = [[] for _ in range(self.clusters)]
        self.best_assignment = None
        self.best_centroids = None
        self.retries = n_retries
        
    def silhouette(self, X: np.ndarray) -> int:
        pass

    def fit_predict(self, X: np.ndarray) -> list[int | np.ndarray]:
        # if self.clusters is None: find  the optimal number of clusters. Need to read up on this.
        
        # center and scale the data if needed
        X = center_scale(X) 
        
        # initialize centroids randomly
        #list of datapoints
        minimum = np.inf
        for trial in range(self.retries):
            self.centroid_assignment = [[] for _ in range(self.clusters)]

            #positions of centroids
            self.centroids = np.zeros(shape=(self.clusters, X.shape[1]))
            for i in range(self.clusters):
                self.centroids[i] += np.random.normal(scale = np.std(X, axis=0) + np.mean(X, axis=0), loc = np.mean(X, axis=0))

            self.last_centroids = np.zeros_like(self.centroids)
            counter = 0
            while counter <= 2:
                while (self.centroids - self.last_centroids).any() != 0:
                    self.update(X)
                else:
                    counter += 1
        
            total = 0
            for i, cluster in enumerate(self.centroid_assignment):
                if len(cluster) < 1:
                    continue
                total += np.sum([l2(self.centroids[i], X[point]) for point in cluster])
            if total < minimum:
                minimum = total
                self.best_assignment = self.centroid_assignment
                self.best_centroids = self.centroids


        # return a list of the clusters for each datpoint (and the positions of the centroids?)
        return self.best_assignment, self.best_centroids

    def update(self, X: np.ndarray) -> None:
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
                    self.centroids[i] = np.hstack([np.mean(X[datapoints][coord], axis=0) for coord in range(X.shape[1])])
                except IndexError:
                    continue
        
    def silhouette(self, X: np.ndarray, mean: bool = True) -> float | np.ndarray:
        """Calculate the Silhouette Coefficient for each sample. or the mean of it"""
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
