import numpy as np
from supers.main import *

    
class MSELoss(Loss):
    
    def __init__(self) -> None:
        super().__init__()

    def get_loss(self, Y: np.ndarray | float, pred: np.ndarray | float) -> np.ndarray | float:
        
        if np.isscalar(Y) and np.isscalar(pred):
            return (pred - Y) ** 2
        
        assert Y.shape[0] == pred.shape[0]
        
        return np.sum([(pred[i] - Y[i]) ** 2 for i in range(len(Y))]) / len(Y)

    def get_grad(self, Y: np.ndarray | float, pred: np.ndarray | float) -> np.ndarray | float:
        
        if np.isscalar(Y) and np.isscalar(pred):
            return 2 * (pred - Y)
        
        assert Y.shape[0] == pred.shape[0]

        return np.sum([2 * (pred[i] - Y[i]) for i in range(len(Y))], axis=1) / len(Y)



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

