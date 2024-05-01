import numpy as np
from parents.parentclasses import *
from utils.maths import diagonal

class ReLU(Layer):

    def __init__(self, position) -> None:
        super().__init__()
        self.pos = position

    def forward(self, X: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(X):
            return max(0, X)
        return np.amax([X, np.zeros(X.shape)], axis = 0)
    
    def grad(self, prev: np.ndarray, X: float | np.ndarray, *args) -> np.ndarray | float:
        if np.isscalar(X):
            return int(X > 0) * prev
        if len(X.shape) < 2:
            X = X.reshape((X.shape[0], 1))
        # assuming n > dim
        return np.dot(prev, diagonal(np.logical_not(X <= 0), is_diagonalizable=True))

    def __str__(self) -> str:
        return f"Activation layer: ReLU_layer, no info_{self.pos}"

class Softmax(Layer):

    #TBD

    def __init__(self, rng: int = None, position: int = 0) -> None:
        super().__init__(rng)
        self.pos = position
    
    def forward(self, X: np.ndarray):
        return np.exp(X) / np.sum(np.exp(X))
    
    def grad(self, prev: np.ndarray, X: float | np.ndarray, *args) -> np.ndarray | float:
        return np.exp(X) / np.sum(np.exp(X))
    
    def __str__(self) -> str:
        return f"Activation layer: Softmax_layer, no info_{self.pos}"
    
class LeakyReLU(Layer):

    def __init__(self, position, alpha=0.1) -> None:
        super().__init__()
        self.pos = position
        self.alpha = alpha

    def forward(self, X: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(X):
            return max(0, X) + self.alpha * min(0, X)
        return np.amax([X, np.zeros(X.shape)], axis = 0) + self.alpha * np.amin([X, np.zeros(X.shape)], axis = 0)
        
    def grad(self, prev: np.ndarray, X: float | np.ndarray, *args) -> np.ndarray | int | float:
        if np.isscalar(X):
            return (1 if X >= 0 else - self.alpha) * prev
        if len(X.shape) < 2:
            X = X.reshape((X.shape[0], 1))
        return np.dot(prev, diagonal((pos := X > 0) - self.alpha * ~pos))
        # This sadly does not work due to too small values leading to rounding errors
        X /= np.amax(X)
        return (ar_3 := - np.amin([((ones := np.ones_like(X)) - (ar_2 := np.amin([np.exp(X), ones], axis = 0))) * (ar_2 + ones), (np.zeros_like(X) + self.alpha)], axis = 0)) + ones + 1 / self.alpha * ones * ar_3
    
    def __str__(self) -> str:
        return f"Activation layer: ReLU_layer, no info_{self.pos}"
    
class Sigmoid(Layer):

    #TBD

    def __init__(self, position) -> None:
        super().__init__()
        self.pos = position

    def forward(self, X: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(X):
            return max(0, X)
        return np.amax([X, np.zeros(X.shape)], axis = 0)
    
    def grad(self, prev: np.ndarray, X: float | np.ndarray, *args) -> np.ndarray | float:
        if np.isscalar(X):
            return int(X > 0)
        if len(X.shape) < 2:
            return np.array([int(num >= 0) for num in X])
        return np.array([[int(num >= 0) for num in x] for x in X]).reshape(X.shape)
    
    def __str__(self) -> str:
        return f"Activation layer: Sigmoid_layer, no info_{self.pos}"
    