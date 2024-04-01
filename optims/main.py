import numpy as np
from supers.main import *

class SGD(optim):

    def __init__(self) -> None:
        super().__init__()
    
    def backpropagation(self) -> None:
        return super().backpropagation()
    
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
