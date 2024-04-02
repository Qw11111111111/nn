"""The parent classes. These should not be called directly."""
import numpy as np
from utils.main import argwhere
import math
class Layer():

    def __init__(self, rng: int = None) -> None:
        """The parent class for Layers. These are passed into models and perform the corresponding calculations.
        Children need to implement a get_grad, a forward, an initialize, a get_state, a load_state, an update_state_dict and a __str__ methos.
        Activation Layers do not necessarily need get_state, load_state, update_state, initialize and update_state_dict methods."""
        if rng:
            self.rng = np.random.RandomState(rng)
        else:
            self.rng = None
        pass

    def get_grad(self, prev: np.ndarray | float, X: np.ndarray | float) -> list[np.ndarray]:
        """returns the gradient of the layer given a previous gradient calculation and an input array X.
        returns the next gradient, the weight gradient and the bias gradient in this order."""
        return prev, 0, 0

    def forward(self, *args):
        return args

    def initialize(self):
        self.bias = np.array(0)
        self.weights = np.array(0)

    def get_state(self):
        pass
    
    def load_state(self):
        pass

    def update_state_dict(self, weight: object, bias: object) -> None:
        pass

    def __str__(self) -> str:
        pass

class Module():

    def __init__(self, layers: list[Layer], rng: int = None, fit_intercept: bool = True) -> None:
        self.layers = layers
        if rng:
            self.rng = np.random.RandomState(rng)
        for layer in self.layers:
            layer.initialize()
        self.fit_intercept = fit_intercept

    def forward(self, x: float | np.ndarray) -> float | np.ndarray:
        if np.isscalar(x):
            for layer in self.layers:
                x = layer.forward(x)
            return x
        x = x.T
        for layer in self.layers:
            x = layer.forward(x)
        return x.T

    def reset(self) -> None:
        for layer in self.layers:
            layer.initialize()

    def get_state_dict(self) -> dict:
        state_dict = {str(layer): {} for layer in self.layers}
        for layer in self.layers:
            if str(layer).endswith("no info"):
                continue
            state_dict.update({str(layer): layer.get_state()})
        
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        for layer in self.layers:
            if str(layer).endswith("no info"):
                continue
            layer.load_state(state_dict[str(layer)])
        
    def backprop_forward(self, X: np.ndarray | float) -> list[np.ndarray | float, str]:
        Y = [X]
        Names = []
        for layer in self.layers:
            Y.append(layer.forward(Y[-1]))
            Names.append(str(layer))
        return Y, Names
    
    def get_grad(self, prev_grad: np.ndarray | float, layer_name: str, X: np.ndarray | float) -> list[np.ndarray]:
        """returns the next gradient, the weight gradient and the bias gradient in this order. Activation layers need only return the next grad for now."""
        return self.layers[argwhere(self.layers, layer_name)].get_grad(prev_grad, X)
        
    def apply_grad(self, weight_grad: dict, bias_grad: dict | None = None) -> None:
        for i, layer in enumerate(self.layers):
            layer.update_state_dict(weight_grad[str(layer)], bias_grad[str(layer)] if self.fit_intercept else None)

class Loss():

    def __init__(self) -> None:
        """The Loss parent class. Children need to implement the get_grad method and the __call__ method."""
        pass
    
    def get_grad(self)-> np.ndarray:
        pass

    def __call__(self, *args: np.any, **kwds: np.any) -> np.any:
        pass

class optim():
    
    def __init__(self, lr: float, model: Module, loss: Loss, stop_val: float = 1e-4) -> None:
        """optimizer parent class. Takes a learning rate, a stop value and performs backpropagation on the given model,
        which needs to implement the methods backprop_forward, get_grad, and apply grad using the given loss class,
        which needs to implement a __call__ and a get_grad method."""
        
        self.lr = lr
        self.model = model
        self.stop_val = stop_val
        self.prev_losses = [np.inf, np.inf]
        self.loss_model = loss

    def backpropagation(self, X: np.ndarray | float, Y: np.ndarray | float) -> None:
        """Performs backpropagation on the model using the Training data X and the Labels Y"""

        FORWARD, NAMES = self.model.backprop_forward(X.T)
        
        loss = self.loss_model(pred=FORWARD[-1], Y=Y.T, axis=1)
        
        if abs((self.prev_losses[-1] - loss)) < self.stop_val or loss > self.prev_losses[1] and self.prev_losses[1] > self.prev_losses[0]:
            return
        
        model_weight_grads = {name: np.zeros(self.model.layers[i].weights.shape) for i, name in enumerate(NAMES)}
        if self.model.fit_intercept:
            model_bias_grads = {name: np.zeros(self.model.layers[i].bias.shape) for i, name in enumerate(NAMES)}
        prev_grad = self.loss_model.get_grad(pred=FORWARD[-1], Y=Y.T)
        
        FORWARD, NAMES = list(reversed(FORWARD)), list(reversed(NAMES))

        for i, name in enumerate(NAMES):
            prev_grad, weight_grad, bias_grad = self.model.get_grad(prev_grad, name, FORWARD[i + 1])
            model_weight_grads[name] += weight_grad
            if self.model.fit_intercept:
                model_bias_grads[name] += np.sum(bias_grad, axis = -1).reshape(model_bias_grads[name].shape)
        
        def normalize(X: np.ndarray) -> float:
            return - 1

        apply_model_weight_grads = {name: 0 for _, name in enumerate(NAMES)}
        if self.model.fit_intercept:
            apply_model_bias_grads = {name: 0 for _, name in enumerate(NAMES)}

        for i, name in enumerate(NAMES):
            apply_model_weight_grads[name] = np.dot(self.lr, np.dot(model_weight_grads[name], normalize(model_weight_grads[name])))
        if self.model.fit_intercept:
            for i, name in enumerate(NAMES):
                apply_model_bias_grads[name] = np.dot(self.lr, np.dot(model_bias_grads[name], normalize(model_bias_grads[name])))

        self.model.apply_grad(apply_model_weight_grads, apply_model_bias_grads if self.model.fit_intercept else None)
        
        self.prev_losses[0], self.prev_losses[1] = self.prev_losses[1], loss


