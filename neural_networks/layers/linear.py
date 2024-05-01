import numpy as np
from parents.parentclasses import Layer

class LinearLayer(Layer):

    def __init__(self, input_dim: int, neurons: int, fit_intercept: bool = True, rng: int = None, position: int = 0) -> None:
        super().__init__(rng)
        self.input_dim = input_dim
        self.neurons = neurons
        self.fit_intercept = fit_intercept
        self.pos = position # automate this
        self.initialize()
    
    def forward(self, X: np.ndarray | float) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
    
    def initialize(self, index: int = None):
        if index:
            self.pos = index
        if self.rng:
            self.weights = self.rng.normal(loc=0, scale= 2 / self.neurons,size=(self.input_dim, self.neurons))
            if self.fit_intercept:
                self.bias = self.rng.normal(loc=0, scale= 2 / self.neurons, size=(1, self.neurons))
            else:
                self.bias = np.zeros((1, self.neurons))
        else:
            self.weights = np.random.normal(loc=0, scale= 2 / self.neurons, size=(self.input_dim, self.neurons))
            if self.fit_intercept:
                self.bias = np.random.normal(loc=0, scale= 2 / self.neurons, size=(1, self.neurons)) # is this correct , or is it actually (1, 1)
            else:
                self.bias = np.zeros((1, self.neurons))
        self.old_weights = self.weights
        self.old_bias = self.bias
        self.weight_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
    

    def get_state(self) -> dict:
        return {"weights": self.weights, "bias": self.bias}

    def load_state(self, state_dict: dict) -> None:
        self.weights, self.bias = state_dict["weights"], state_dict["bias"]
    
    def update_state_dict(self, weight_update: np.ndarray, bias_update: np.ndarray | None = None) -> None:
        self.old_weights, self.weights = self.weights, self.weights + weight_update
        if self.fit_intercept:
            self.old_bias, self.bias = self.bias, self.bias + bias_update

    def grad(self, prev: np.ndarray, X: np.ndarray) -> np.ndarray:
        """returns the next grad and applies the weight and bias grad to self"""
        next_grad, self.weight_grad, self.bias_grad = np.dot(prev, self.weights.T), np.dot(X.T, prev), prev if self.fit_intercept else self.bias
        assert self.weight_grad.shape == self.weights.shape, f"wrong weight grad shape in layer {str(self)}, grad: {self.weight_grad.shape}, weight: {self.weights.shape}"
        assert self.bias_grad.shape == self.bias.shape, f"wrong bias grad shape in layer {str(self)}"
        return next_grad

    def __str__(self) -> str:
        return f"Linear_layer_{self.pos}"
    
