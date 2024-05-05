"""The parent classes. These should not be called directly."""
from socket import NI_NAMEREQD
import numpy as np

class Layer():
    
    def __init__(self, rng: int = None, pos: int = 0, *args, **kwargs) -> None:
        """The parent class for Layers. These are passed into models and perform the corresponding calculations.
        Children need to implement a get_grad, a forward, an initialize, a get_state, a load_state, an update_state_dict and a __str__ methos.
        Activation Layers do not necessarily need get_state, load_state, update_state, initialize and update_state_dict methods."""
        if rng:
            self.rng = np.random.RandomState(rng)
        else:
            self.rng = None
        self.pos = pos

    '''def get_grad(self, prev: np.ndarray | float, X: np.ndarray | float) -> list[np.ndarray]:
        """returns the gradient of the layer given a previous gradient calculation and an input array X.
        returns the next gradient, the weight gradient and the bias gradient in this order."""
        return prev, 0, 0'''

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X

    def initialize(self, index: int = None) -> None:
        if index:
            self.pos = index
        self.bias = np.array(0)
        self.weights = np.array(0)
        self.old_bias = self.bias
        self.old_weights = self.weights
        self.bias_grad = np.zeros_like(self.bias)
        self.weight_grad = np.zeros_like(self.weights)

    def get_state(self) -> dict[str, np.ndarray]:
        pass
    
    def load_state(self, state_dict: dict[str, np.ndarray]) -> None:
        pass

    def update_state_dict(self, weight: np.ndarray, bias: np.ndarray) -> None:
        pass

    def grad(self, prev: np.ndarray, X: np.ndarray) -> np.ndarray:
        """returns the next grad and applies the weight and bias grad to self"""
        next_grad = prev
        return next_grad

    def __str__(self) -> str:
        return f"Layer at position_{self.pos}"

class Module():

    def __init__(self, layers: list[Layer], rng: int = None, fit_intercept: bool = True, layer_order: str = "auto", *args, **kwargs) -> None:
        self.layers = layers
        if rng:
            self.rng = np.random.RandomState(rng)
        for i, layer in enumerate(self.layers):
            position = i if layer_order == "auto" else None
            layer.initialize(index=position)
        self.fit_intercept = fit_intercept
        self.best_state_dict = self.get_state_dict()
        self.dropouts = np.zeros(len(self.layers))

    def forward(self, x: float | np.ndarray) -> float | np.ndarray:
        if np.isscalar(x):
            for layer in self.layers:
                x = layer.forward(x)
            return x
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def reset(self) -> None:
        for layer in self.layers:
            layer.initialize()

    def get_state_dict(self) -> dict[str, np.ndarray]:
        state_dict = {str(layer): {} for layer in self.layers}
        for layer in self.layers:
            if str(layer).endswith("no info"):
                continue
            state_dict.update({str(layer): layer.get_state()})
        
        return state_dict

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        for layer in self.layers:
            if str(layer).endswith("no info"):
                continue
            layer.load_state(state_dict[str(layer)])
        
    def backprop_forward(self, X: np.ndarray | float) -> list[np.ndarray, str]:
        Y = [X]
        Names = []
        for layer in self.layers:
            Y.append(layer.forward(Y[-1]))
            Names.append(str(layer))
        return Y, Names
    
    '''def get_grad(self, prev_grad: np.ndarray | float, layer_name: str, X: np.ndarray | float) -> list[np.ndarray]:
        """returns the next gradient, the weight gradient and the bias gradient in this order. Activation layers need only return the next grad for now."""
        return self.layers[argwhere(self.layers, layer_name)].get_grad(prev_grad, X)'''
        
    def apply_grad(self, weight_grad: dict[str, np.ndarray], bias_grad: dict[str, np.ndarray] | None = None) -> None:
        for i, layer in enumerate(self.layers):
            if self.dropouts[i] == 1:
                pass
            layer.update_state_dict(weight_grad[str(layer)], bias_grad[str(layer)] if self.fit_intercept else None)
        
    def set_params(self, *args, **kwargs) -> None:
        pass

    def dropout(self, dropout_rate: float = 2e-1, *arg, **kwargs) -> np.any:
        for i, layer in enumerate(self.layers):
            if np.random.random() < dropout_rate:
                self.dropouts[i] = 1


class Loss():

    def __init__(self, *args, **kwargs) -> None:
        """The Loss parent class. Children need to implement the get_grad method and the __call__ method."""
        pass
    
    def get_grad(self, Y: np.ndarray | float, pred: np.ndarray | float)-> np.ndarray:
        """returns an array of shape(pred.shape)"""
        pass

    def __call__(self, Y: np.ndarray | float, pred: np.ndarray | float) -> float:
        pass

class optim():
    
    def __init__(self, lr: float, model: Module, loss: Loss, stop_val: float = 1e-4, *args, **kwargs) -> None:
        """optimizer parent class. Takes a learning rate, a stop value and performs backpropagation on the given model,
        which needs to implement the methods backprop_forward, get_grad, and apply grad using the given loss class,
        which needs to implement a __call__ and a get_grad method."""
        
        self.lr = lr
        self.model = model
        self.stop_val = stop_val
        self.prev_losses = [np.inf, np.inf]
        self.loss_model = loss

    def backpropagation(self, X: np.ndarray, Y: np.ndarray) -> None:
        """performs one optimization step over all provided data points"""
        FORWARD, NAMES = self.model.backprop_forward(X)
        self.batch_size = X.shape[0]
    	
        FORWARD, NAMES = list(reversed(FORWARD)), list(reversed(NAMES))

        loss = self.loss_model(pred=FORWARD[0], Y=Y)
        
        if abs((self.prev_losses[-1] - loss)) < self.stop_val or loss > self.prev_losses[1] and self.prev_losses[1] > self.prev_losses[0]:
            pass

        prev_grad = self.loss_model.get_grad(pred=FORWARD[-1], Y=Y)
        
        for i, layer in enumerate(reversed(self.model.layers)):
            prev_grad = layer.grad(prev_grad, FORWARD[i + 1])

        self.prev_losses[0], self.prev_losses[1] = self.prev_losses[1], loss
    
    def set_params(self, *args, **kwargs) -> None:
        self.model = kwargs["model"]
        self.lr = kwargs["lr"]
        self.loss_model = kwargs["loss_model"]

class Clusterer():

    def __init__(self, clusters: int | None = None) -> None:
        self.n_clusters = clusters

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def _update(self, X: np.ndarray) -> None:
        pass
