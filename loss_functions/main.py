import numpy as np
from supers.main import *
    
class MSELoss(Loss):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def get_grad(self, Y: np.ndarray | float, pred: np.ndarray | float, axis: int | bool = 1) -> np.ndarray | float:
        if np.isscalar(Y) and np.isscalar(pred):
            return (pred - Y)
        
        grad = pred - Y
        assert grad.shape == pred.shape, f"wrong shape: {grad.shape}"
        return grad
    
    def __call__(self, Y: np.ndarray | float, pred: np.ndarray | float, axis: int = 0) -> np.ndarray | float:
        if np.isscalar(Y) and np.isscalar(pred):
            return (pred - Y) ** 2
        
        return np.sum(np.square(np.sum([pred, - Y], axis = 0))) / Y.shape[axis]

class RMSELoss(MSELoss):

    #TBD: get grad

    def __init__(self) -> None:
        super().__init__()
    
    def get_grad(self, Y: np.ndarray | float, pred: np.ndarray | float, axis: int | bool = 1) -> np.ndarray | float:
        return (super().get_grad(Y, pred, axis))
    
    def __call__(self, Y: np.ndarray | float, pred: np.ndarray | float, axis: int = 0) -> np.ndarray | float:
        return np.sqrt(super().__call__(Y, pred, axis))

class  MAELoss(Loss):

    #TBD: get grad

    def __init__(self) -> None:
        super().__init__()

    def get_grad(self, Y: np.ndarray | float, pred: np.ndarray | float, axis: int | bool = 1) -> np.ndarray:
        if np.isscalar(Y) and np.isscalar(pred):
            return pred
        return ((np.sum(np.sum([pred, - Y], axis = 0), axis = int(not axis))) / Y.shape[int(axis)]).reshape(1, Y.shape[1])
    
    def __call__(self, pred: np.ndarray, Y: np.ndarray, axis: int = 0) -> np.any:
        if np.isscalar(Y) and np.isscalar(pred):
            return pred - Y
        return np.sum(np.sum([pred, - Y], axis = 0)) / Y.shape[axis]

class CrossEntropyLoss(Loss):

    #TODO: all

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, Y: np.ndarray | float, pred: np.ndarray | float) -> float:
        return super().__call__(Y, pred)
    
    def get_grad(self, Y: np.ndarray | float, pred: np.ndarray | float) -> np.ndarray:
        return super().get_grad(Y, pred)

    
