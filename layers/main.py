import numpy as np
from supers.main import *

class LinearLayer(Layer):

    # Need to transpose all layers in order to allow inputs of shape(n, dim)

    def __init__(self, input_dim: int, neurons: int, fit_intercept: bool = True, rng: int = None, position: int = 0) -> None:
        super().__init__(rng)
        self.input_dim = input_dim
        self.neurons = neurons
        self.fit_intercept = fit_intercept
        self.pos = position # automate this
        self.initialize()

    def forward(self, X: np.ndarray | float) -> np.ndarray:
        return np.dot(self.weights, X) + self.bias
    
    def get_grad(self, prev: np.ndarray | float, X: np.ndarray | float) -> list[np.ndarray, float]:
        #assert prev.shape[0] == X.T.shape[1] or np.isscalar(X)
        return np.dot(self.weights.T, prev), np.dot(prev, X.T), prev if self.fit_intercept else 0
    
    def initialize(self, index: int = None):
        if index:
            self.pos = index
        if self.rng:
            self.weights = self.rng.random((self.neurons, self.input_dim)) - 0.5
            if self.fit_intercept:
                self.bias = self.rng.random((self.neurons, 1)) - 0.5
            else:
                self.bias = np.zeros((self.neurons, 1))
        else:
            self.weights = np.random.random((self.neurons, self.input_dim)) - 0.5
            if self.fit_intercept:
                self.bias = np.random.random((self.neurons, 1)) - 0.5
            else:
                self.bias = np.zeros((self.neurons, 1))
        self.old_weights = self.weights
        self.old_bias = self.bias
            
    def get_state(self) -> dict:
        return {"weights": self.weights, "bias": self.bias}

    def load_state(self, state_dict: dict) -> None:
        self.weights, self.bias = state_dict["weights"], state_dict["bias"]
    
    def __str__(self) -> str:
        return f"Linear_layer_{self.pos}"
    
    def update_state_dict(self, weight_update: np.ndarray, bias_update: np.ndarray | None = None) -> None:
        self.old_weights, self.weights = self.weights, self.weights + weight_update
        if self.fit_intercept:
            self.old_bias, self.bias = self.bias, self.bias + bias_update

class ReLU(Layer):

    def __init__(self, position) -> None:
        super().__init__()
        self.pos = position

    def forward(self, X: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(X):
            return max(0, X)
        return np.amax([X, np.zeros(X.shape)], axis = 0)
    
    def get_grad(self, _: np.any, X: float | np.ndarray, *args) -> np.ndarray | int:
        if np.isscalar(X):
            return int(X > 0), 0, 0
        if len(X.shape) < 2:
            return np.array([int(num >= 0) for num in X]), 0, 0
        return np.array([[int(num >= 0) for num in x] for x in X]).reshape(X.shape), 0, 0
    
    def __str__(self) -> str:
        return f"Activation layer: ReLU_layer, no info_{self.pos}"

class  Softmax(Layer):

    #TBD

    def __init__(self, rng: int = None, position: int = 0) -> None:
        super().__init__(rng)
        self.pos = position
    
    def forward(self, X: np.ndarray):
        return np.exp(X) / np.sum(np.exp(X))
        return super().forward(*args)
    
    def get_grad(self, prev: np.ndarray | float, X: np.ndarray | float) -> list[np.ndarray]:
        return np.exp(X) / np.sum(np.exp(X))
        return super().get_grad(prev, X)
    
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
    
    def get_grad(self, _: np.any, X: float | np.ndarray, *args) -> np.ndarray | int:
        if np.isscalar(X):
            return 1 if X >= 0 else - self.alpha
        if len(X.shape) < 2:
            return np.array([1 if num >= 0 else - self.alpha for num in X])
        return np.array([[1 if num >= 0 else - self.alpha for num in row] for row in X]), 0, 0
        # This sadly does not work due to too small values leading to rounding errors
        X /= np.amax(X)
        return (ar_3 := - np.amin([((ones := np.ones_like(X)) - (ar_2 := np.amin([np.exp(X), ones], axis = 0))) * (ar_2 + ones), (np.zeros_like(X) + self.alpha)], axis = 0)) + ones + 1 / self.alpha * ones * ar_3, 0, 0
        
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
    
    def get_grad(self, _: np.any, X: float | np.ndarray, *args) -> np.ndarray | int:
        if np.isscalar(X):
            return int(X > 0), 0, 0
        if len(X.shape) < 2:
            return np.array([int(num >= 0) for num in X]), 0, 0
        return np.array([[int(num >= 0) for num in x] for x in X]).reshape(X.shape), 0, 0
    
    def __str__(self) -> str:
        return f"Activation layer: Sigmoid_layer, no info_{self.pos}"
    

class Flatten(Layer):

    #TBD
    
    def __init__(self, position) -> None:
        super().__init__()
        self.pos = position

    def forward(self, X: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(X):
            return X
        return np.reshape(X, -1)
    
    def get_grad(self, _: np.any, X: float | np.ndarray, *args) -> np.ndarray | int:
        if np.isscalar(X):
            return X, 0, 0
        if len(X.shape) < 2:
            return X, 0, 0
        return X, 0, 0
    
    def __str__(self) -> str:
        return f"Activation layer: Flatten_layer, no info_{self.pos}"

class Conv1dLayer(Layer):
    pass

class Conv2dLayer(Layer):
    pass

class MaxPool1dLayer(Layer):
    pass

class MaxPool2dLayer(Layer):
    pass
