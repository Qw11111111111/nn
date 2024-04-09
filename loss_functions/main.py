import numpy as np
from supers.main import *

    
class MSELoss(Loss):
    
    def __init__(self) -> None:
        super().__init__()

    def get_grad(self, Y: np.ndarray | float, pred: np.ndarray | float, axis: int | bool = 1) -> np.ndarray | float:
        if np.isscalar(Y) and np.isscalar(pred):
            return 2 * (pred - Y)
        
        #assert Y.shape[0] == pred.shape[0]
        #print(Y, pred)
        return (np.sum(np.dot(2, np.sum([pred, - Y], axis = 0)), axis = int(not axis)) / Y.shape[int(axis)]).reshape(1, Y.shape[1])
        return np.sum([2 * (pred[i] - Y[i]) for i in range(len(Y))], axis=1) / len(Y)
    
    def __call__(self, Y: np.ndarray | float, pred: np.ndarray | float, axis: int = 0) -> np.ndarray | float:
        if np.isscalar(Y) and np.isscalar(pred):
            return (pred - Y) ** 2
        #assert Y.shape[0] == pred.shape[0]
        
        return np.sum(np.square(np.sum([pred, - Y], axis = 0))) / Y.shape[axis]
        return np.sum([(pred[i] - Y[i]) ** 2 for i in range(len(Y))]) / len(Y)

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

    
