from colorsys import yiq_to_rgb
import numpy as np
from supers.main import *

class SGD(optim):

    def __init__(self, batch_size: int = 1, momentum: bool = False, alpha: float = 0.01) -> None:
        super().__init__()
        self.batch_size = batch_size
    
    def backpropagation(self, X: np.ndarray | float, Y: np.ndarray | float) -> None:
        random_perm = np.random.permutation(len(Y))
        X, Y = X[random_perm], Y[random_perm]
        for i in range(int(len(X) / self.batch_size) - 1):
            X_train, Y_train = X[i:i + self.batch_size], Y[i:i + self.batch_size]
            super().backpropagation(X_train, Y_train)
        
    
class Adam(optim):

    def __init__(self) -> None:
        super().__init__()

    def backpropagation(self) -> None:
        return super().backpropagation()
    
class Momentum(optim):

    def __init__(self) -> None:
        super().__init__()

    def backpropagation(self) -> None:
        return super().backpropagation()
    
class GD(optim):

    def __init__(self, lr: float, model: Module, loss: Loss, stop_val: float = 1e-4) -> None:
        super().__init__(lr, model, loss, stop_val)

    def backpropagation(self, X: np.ndarray, Y: np.ndarray) -> None:
        return super().backpropagation(X, Y)
