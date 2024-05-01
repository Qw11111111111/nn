from parents.parentclasses import Layer
import numpy as np

class Flatten(Layer):

    #TBD
    
    def __init__(self, position) -> None:
        super().__init__()
        self.pos = position

    def forward(self, X: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(X):
            return X
        return np.reshape(X, -1)
    
    def grad(self, prev: np.ndarray, X: float | np.ndarray, *args) -> np.ndarray | float:
        if np.isscalar(X):
            return X
        if len(X.shape) < 2:
            return X
        return X
    
    def __str__(self) -> str:
        return f"Activation layer: Flatten_layer, no info_{self.pos}"
