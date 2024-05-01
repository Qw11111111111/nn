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