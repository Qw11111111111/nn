import numpy as np
from supers.main import *

class LinearLayer(Layer):

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
    
    def initialize(self):
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
            
    def get_state(self) -> dict:
        return {"weights": self.weights, "bias": self.bias}

    def load_state(self, state_dict: dict) -> None:
        self.weights, self.bias = state_dict["weights"], state_dict["bias"]
    
    def __str__(self) -> str:
        return f"Linear_layer_{self.pos}"
    
    def update_state_dict(self, weight_update: np.ndarray, bias_update: np.ndarray | None = None) -> None:
        self.weights += weight_update
        if self.fit_intercept:
            self.bias += bias_update.reshape(self.bias.shape)

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
            return int(X > 0)
        if len(X.shape) < 2:
            return np.array([int(num >= 0) for num in X])
        return np.array([[int(num >= 0) for num in x] for x in X]).reshape(X.shape), 0, 0
    
    def __str__(self) -> str:
        return f"Activation layer: ReLU_layer, no info_{self.pos}"

