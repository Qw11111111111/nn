import numpy as np
from parents.parentclasses import Loss

class MSELoss(Loss):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def get_grad(self, Y: np.ndarray | float, pred: np.ndarray | float, axis: int | bool = 1) -> np.ndarray | float:
        if np.isscalar(Y) and np.isscalar(pred):
            return (pred - Y) / pred.shape[0]
        
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
#https://en.wikipedia.org/wiki/Cross-entropy
    #TODO: check if correct

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, Y: np.ndarray | float, pred: np.ndarray | float) -> float:
        return np.sum(np.dot(-Y.T, np.log(pred)) - np.dot((1 - Y).T, np.log(1 - pred)))
    
    def get_grad(self, Y: np.ndarray | float, pred: np.ndarray | float) -> np.ndarray:
        if Y.shape != pred.shape:
            Y = ~np.bool_(Y)
            Y = np.column_stack((Y, ~Y))
            Y = np.int64(Y)
        if np.isscalar(Y) and np.isscalar(pred):
            return (pred - Y) / pred.shape[0]
        
        grad = pred - Y
        assert grad.shape == Y.shape, f"wrong shape: {grad.shape}"
        return grad

    
