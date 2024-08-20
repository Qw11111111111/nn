from typing import Any
from abc import abstractmethod, ABC
import jax.numpy as jnp

class Layer(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    
    @abstractmethod
    def __call__(self, X: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def backward(self, X: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def initialize(self):
        pass