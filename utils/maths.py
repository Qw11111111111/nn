import numpy as np

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
        if not y:
            return np.sum(np.square(w))
        else:
            return np.sum(np.square(w - y))
    else:
        if not y:
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
    
    def __init__(self, num_clusters: int | None = None) -> None:
        self.clusters = num_clusters

    def fit_predict(self, X: np.ndarray) -> list[int]:
        # if self.clusters is None: find  the optimal number of clusters. Need to read up on this.
        
        # initialize centroids randomly

        # assign each point to a cluster based on their closest centroid

        # update the centroids using the average of all points assigned to it as a new centroid

        # assign  all points to the cluster with the smallest distance to its centroid and repeat until no more changes can be made.
        
        # return a list of the clusters for each datpoint (and the positions of the centroids?)
        pass