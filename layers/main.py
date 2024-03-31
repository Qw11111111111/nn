import numpy as np

class Layer():

    def __init__(self, rng: int) -> None:
        if rng:
            self.rng = np.random.RandomState(rng)
        else:
            self.rng = None
        pass

    def get_grad(self):
        pass

    def forward(self):
        pass

    def initialize():
        pass

    def get_state(self):
        pass
    
    def load_state(self):
        pass

class LinearLayer(Layer):

    def __init__(self, input_dim: int, neurons: int, output_dim: int = 1, fit_intercept: bool = True, rng: int = None) -> None:
        super().__init__(rng)
        self.input_dim = input_dim
        self.neurons = neurons
        self.output_dim = output_dim
        self.fit_intercept = fit_intercept
        self.initialize()

    def forward(self, X: np.ndarray | float) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
    
    def get_grad(self, prev: np.ndarray | float, X: np.ndarray | float) -> list[np.ndarray, float]:
        assert prev.shape[1] == X.T.shape[0] or np.isscalar(X)
        return np.dot(prev, X.T), prev if self.fit_intercept else 0
    
    def initialize(self):
        if self.rng:
            self.weights = self.rng.random((self.input_dim, self.neurons))
            if self.fit_intercept:
                self.bias = self.rng.random((1, self.neurons))
            else:
                self.bias = np.zeros((1, self.neurons))
        else:
            self.weights = np.random.random((self.input_dim, self.neurons))
            if self.fit_intercept:
                self.bias = np.random.random((1, self.neurons))
            else:
                self.bias = np.zeros((1, self.neurons))
            
    def get_state(self) -> dict:
        return {"weights": self.weights, "bias": self.bias}

    def load_state(self, state_dict: dict) -> None:
        self.weights, self.bias = state_dict["weights"], state_dict["bias"]

    
