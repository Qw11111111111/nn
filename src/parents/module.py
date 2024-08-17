from typing import Any
from abc import ABC, abstractmethod
import jax.numpy as jnp
from parents.optim import Optim
from parents.loss import Loss

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

    def fit(self, optim: Optim, criterion: Loss, X: jnp.ndarray, Y: jnp.ndarray, *args: Any, **kwargs: Any) -> None:
        pass

    def initialize(self) -> None:
        pass

    def get_state(self) -> dict:
        pass

    def apply_state(self, state: dict) -> None:
        pass