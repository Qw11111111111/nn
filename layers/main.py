import numpy as np

class Layer():

    def __init__(self, rng: int = None) -> None:
        if rng:
            self.rng = np.random.RandomState(rng)
        else:
            self.rng = None
        pass

    def get_grad(self):
        pass

    def forward(self, *args):
        return args

    def initialize():
        pass

    def get_state(self):
        pass
    
    def load_state(self):
        pass

    def __str__(self) -> str:
        pass

class LinearLayer(Layer):

    def __init__(self, input_dim: int, neurons: int, output_dim: int = 1, fit_intercept: bool = True, rng: int = None, position: int = 0) -> None:
        super().__init__(rng)
        self.input_dim = input_dim
        self.neurons = neurons
        self.output_dim = output_dim
        self.fit_intercept = fit_intercept
        self.pos = position
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
    
    def __str__(self) -> str:
        return f"Linear_layer_{self.pos}"

class ReLU(Layer):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: np.ndarray | float) -> np.ndarray | float:
        if np.isscalar(X):
            return max(0, X)
        if len(X.shape) < 2:
            X = X.reshape((-1, X.shape[0]))
        return np.array([[np.maximum(0, num) for num in x] for x in X])
    
    def get_grad(self, X: float | np.ndarray) -> np.ndarray | int:
        if np.isscalar(X):
            return int(X > 0)
        if len(X.shape) < 2:
            X = X.reshape((-1, X.shape[0]))
        return  np.array([[int(num > 0) for num in x] for x in X])
    
    def __str__(self) -> str:
        return f"ReLU_layer, no info"

