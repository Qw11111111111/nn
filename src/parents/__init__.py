from typing import Any
from abc import abstractmethod, ABC
import jax.numpy as jnp

class Clusterer(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        pass

    @abstractmethod
    def fit_predict(self, X: jnp.ndarray, *args: Any, **kwds: Any) -> jnp.ndarray:
        pass

    def set_params(self, *args: Any, **kwargs: Any) -> None:
        pass
    
class Layer(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        pass
    
    @abstractmethod
    def __call__(self, X: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass

    def backward(self, X: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def initialize(self):
        pass
    
    def get_state(self) -> dict:
        pass
    
    def apply_state(self, state: dict) -> None:
        pass
    
class Loss(ABC):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        pass

    @abstractmethod
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray, *args: Any, **kwds: Any) -> jnp.ndarray:
        pass
    
    def backward(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        pass

class Module(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        pass

    @abstractmethod
    def __call__(self, X: jnp.ndarray, *args: Any, **kwds: Any) -> jnp.ndarray:
        pass

    def __str__(self) -> str:
        pass

    def initialize(self) -> None:
        pass

    def get_state(self) -> dict:
        pass

    def apply_state(self, state: dict) -> None:
        pass
    
class Optim(ABC):

    @abstractmethod
    def __init__(self, model: Module, lr: float, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.lr = lr
        self.model = model

    def step(self, loss_grad: jnp.ndarray) -> Any | None:
        pass
    
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