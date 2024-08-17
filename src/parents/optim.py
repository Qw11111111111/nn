from typing import Any
from abc import ABC, abstractmethod
import jax.numpy as jnp

class Optim(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def backward(self) -> Any | None:
        pass
    
    def set_params(self, *args: Any, **kwargs: Any) -> None:
        pass