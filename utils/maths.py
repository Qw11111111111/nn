import numpy as np
from utils.utils import argwhere, timeit

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
    
def center_scale(X: np.ndarray, axis: int = 0, verbose: bool = False) -> np.ndarray:
    """returns the centered and scaled data, as well as the mean and std if set to verbose"""
    mean = np.mean(X, axis = axis)
    std = np.std(X, axis=axis)
    if verbose:
        return (X - mean) / std, mean, std
    return (X - mean) / std

@timeit
def PCA(X: np.ndarray, n_components: int | None = None, axis: int = 0) -> np.ndarray:
    # center and scale the data 
    X = center_scale(X, axis)
    U, S, V = np.linalg.svd(X)
    # truncate the result
    #S = np.diag(S) should be done after pivot search i think
    if n_components is None:
        # find optimal split https://stackoverflow.com/questions/4471993/compute-the-elbow-for-a-curve-automatically-and-mathematically
        # i belive that this does not currently work correctly
        pivot = np.argmax([S[i + 1] + S[i - 1] - 2 * S[i] for i in range(1, len(S) - 1)])
    else:
        pivot = n_components

    #default to 1 if auto calculates something false
    if pivot < 1:
        pivot = 1
    S = np.diag(S)
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

def kernel_PCA(X: np.ndarray, kernel: object) -> np.ndarray:
    #TODO: implement
    M = np.zeros(shape=(X.shape[1], X.shape[1]))
    # Mij = K(xi, xj)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i, j] = kernel(X[i][j], X[j][i]) #this is gonna raise an indexerror...

def gaussian_kernel(X: np.ndarray | float, Y: np.ndarray | float, epsilon: float = 1e-1) -> np.ndarray | float:
    if np.isscalar(X):
        return np.exp(-l2(X, Y, squared=True) / np.square(epsilon))

def mean_of_cluster(X) -> float:
    return np.hstack([np.mean(X[:,coord], axis=0) for coord in range(X.shape[1])])

@timeit
def matmul(X, Y):
    return np.matmul(X, Y)

@timeit
def dot(X, Y):
    return np.dot(X, Y)

@timeit
def appr_silhouette(self, X: np.ndarray, centroids: np.ndarray, clusters: np.ndarray | list, mean: bool = True) -> float | np.ndarray:
        """Apply the Silhouette method on an unsupervised learning model."""
        silhouette_scores = np.zeros(shape=X.shape[0])
        for i, point in enumerate(X):
            minimum = np.inf
            assignment = argwhere(clusters, i, axis=1)[0]
            a = l2(point, centroids[assignment])
            b = np.min([l2(point, centroid) if j != assignment else np.inf for j, centroid in enumerate(centroids)])
            try:
                silhouette_scores[i] = ((b - a) / (np.amax([a, b]))) if np.amax([a, b]) != 0 else 1 if np.amax([a, b]) != np.inf else 0 if np.amax([a, b]) != np.nan else 0.5 #this still raises and shows a runtime warning. Need to fix
            except RuntimeWarning:
                silhouette_scores[i] = 1 if np.amax([a, b]) == 0 else 0
        return np.mean(silhouette_scores) if mean else silhouette_scores

