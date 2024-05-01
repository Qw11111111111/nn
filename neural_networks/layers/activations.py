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
        return np.dot(prev, diagonal(X > 0, is_diagonalizable=True))

    def __str__(self) -> str:
        return f"Activation layer: ReLU_layer, no info_{self.pos}"

class Softmax(Layer):

    #TBD

    def __init__(self, rng: int = None, position: int = 0) -> None:
        super().__init__(rng, position)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.array([np.exp(x) / np.sum(np.exp(x), axis=0) for x in X])
    
    def grad(self, prev: np.ndarray, X: float | np.ndarray, *args) -> np.ndarray:
        # problem: X is of shape n*m, but softmax wants 1*m with the output being m*m. --> sum up all jacobians for each vector 
        #is this true, or should this actually be diag(softmax(X))?
        #--> need to loop over data ig...
        jacobian = np.zeros(shape = (X.shape[1], X.shape[1]))
        #which = np.bool_(np.diag(np.ones(shape=(X.shape[1]))))
        for x in X:
            jacobian += np.diag([x[i] * (1 - x[i]) for i in range(len(x))])
            jacobian += np.array([[-x[i] * x[j] if i == j else 0 for j in range(len(x))] for i in range(len(x))])

        return np.dot(prev, jacobian) # do i need to take diagonal() on jac again?
    
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
    