from typing import Any
from abc import abstractmethod, ABC
import jax.numpy as jnp
import numpy as np
import jax

class Clusterer(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        pass

    @abstractmethod
    def fit_predict(self, X: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        pass

    def set_params(self, *args: Any, **kwargs: Any) -> None:
        pass
    
class Layer(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.initialize()
    
    @abstractmethod
    def __call__(self, X: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        pass
    
    def __str__(self) -> str:
        return str(self.__class__).split('.')[-1][:-2]
        

    def backward(self, X: jnp.ndarray) -> jnp.ndarray:
        return jax.jacobian(self)(X)

    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def get_state(self) -> tuple:
        return ()
    
    @abstractmethod
    def apply_state(self, state: tuple) -> None:
        pass
    
class Loss(ABC):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        pass

    @abstractmethod
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        pass
    
    def backward(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(self)(X, Y)

class Module(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.layers: list[Layer]
        #self.initialize()

    @abstractmethod
    def __call__(self, X: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        pass

    def __str__(self) -> str:
        string_repr = str(self.__class__).split('.')[-1][:-2] + "\n"
        for layer in self.layers:
            string_repr += "\t-" + str(layer) + "\n"
        return string_repr
        
    def initialize(self) -> None:
        for layer in self.layers:
            layer.initialize()

    def get_state(self) -> dict:
        return {str(i) + ',' + str(layer): layer.get_state() for i, layer in self.layers}

    def apply_state(self, state: dict) -> None:
        for i, layer in self.layers:
            key = str(i) + ',' + str(layer)
            layer.apply_state(state[key])
    
    def _forward_verbose(self, X: jnp.ndarray) -> list[jnp.ndarray]:
        forward = [X]
        for layer in self.layers:
            forward.append(layer(forward[-1]))
        return forward
    
class Optim(ABC):

    #@abstractmethod
    def __init__(self, model: Module, criterion: Loss, lr: float = 1e-4, momentum: int = 1e-1, rng: int | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.lr = lr
        self.model = model
        self.criterion = criterion
        self.rng = np.random.RandomState(rng) if rng is not None else rng
        self.momentum = momentum
        self.last_weight_update, self.last_bias_update = [0 for _ in range(len(self.model.layers))], [0 for _ in range(len(self.model.layers))]

    @abstractmethod
    def step(self, X: jnp.ndarray, Y: jnp.ndarray) -> None:
        FORWARD = self.model._forward_verbose(X)
        loss_grad = self.criterion.backward(FORWARD[-1], Y)
        for i, layer in enumerate(reversed(self.model.layers)):
            layer_grad_w = jnp.dot(loss_grad, layer.backward(FORWARD[-i - 2])[0]) #TODO: correct these to correctly use the generated jacobian # get d_layer/d_weights
            layer_grad_b = jnp.dot(loss_grad, layer.backward(FORWARD[-i - 2])[1])                                                              # get d_layer/d_bias
            self.last_weight_update[i] = -self.lr * layer_grad_w.T + self.momentum * self.last_weight_update[i]
            self.last_bias_update[i] = -self.lr * layer_grad_b.T.sum(axis = 0) + self.momentum * self.last_bias_update[i]
            layer.weights += self.last_weight_update[i]
            layer.bias += self.last_bias_update[i]
            loss_grad = loss_grad.dot(layer.backward(X[-i - 2]))                                                                               # get d_layer/d_prev_layer
    
    @abstractmethod
    def set_params(self, *args: Any, **kwargs: Any) -> None:
        pass
    
__all__ = [
    "Module",
    "Optim",
    "Clusterer",
    "Loss",
    "Layer"
]