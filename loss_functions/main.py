import numpy as np
from numpy.core.fromnumeric import any as any
from supers.main import *

    
class MSELoss(Loss):
    
    def __init__(self) -> None:
        super().__init__()

    def get_grad(self, Y: np.ndarray | float, pred: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(Y) and np.isscalar(pred):
            return 2 * (pred - Y)
        
        assert Y.shape[0] == pred.shape[0]

        return np.sum([2 * (pred[i] - Y[i]) for i in range(len(Y))], axis=1) / len(Y)
    
    def __call__(self, Y: np.ndarray | float, pred: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(Y) and np.isscalar(pred):
            return (pred - Y) ** 2
        
        assert Y.shape[0] == pred.shape[0]
        
        return np.sum([(pred[i] - Y[i]) ** 2 for i in range(len(Y))]) / len(Y)




