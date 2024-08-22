from typing import Any
from abc import ABC, abstractmethod
import jax.numpy as jnp

class Module(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(self, X: jnp.ndarray, *args: Any, **kwds: Any) -> jnp.ndarray:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def fit(self, criterion, X: jnp.ndarray, Y: jnp.ndarray, optim, *args: Any, **kwargs: Any) -> None:
        
        pass

    def initialize(self) -> None:
        pass

    def get_state(self) -> dict:
        pass

    def apply_state(self, state: dict) -> None:
        pass