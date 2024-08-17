from typing import Any
from abc import abstractmethod, ABC
import jax.numpy as jnp

class Loss(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray, *args: Any, **kwds: Any) -> jnp.ndarray:
        pass
    
    @abstractmethod
    def backward(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        pass
