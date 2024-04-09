import numpy as np
from sympy import per

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

def argwhere(v: list | np.ndarray, val: str | float) -> int:
    for i, value in enumerate(v):
        if str(value) == str(val):
            return i

class CV():
    def __init__(self, cv_folds: int = 5, permute: bool = True):
        self.cv_folds = cv_folds
        self.permute = permute

    def get_indices(self, X: np.ndarray):
        if self.permute:
            permutation = np.random.permutation(len(X))
        else:
            permutation = np.arange(len(X))
        fold_size = int(len(permutation) / self.cv_folds)
        indices = [permutation[i * fold_size:i * fold_size + fold_size] for i in range(self.cv_folds)]
        print(indices)
        for i in range(self.cv_folds):
            yield indices[:i] + indices[i + 1:], indices[i]
    