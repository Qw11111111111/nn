from src.parents import Layer
from src import LinearLayer, ReLU
from src.parents import Module
from src.parents import Loss
from src.parents import Optim
import jax.numpy as jnp
import jax
import numpy as np
from typing import Any


class MLP(Module): 

    def __init__(self, shape: tuple[int, int], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layers = [
            LinearLayer((shape[0], 2), False),
            ReLU(),
            LinearLayer((2, shape[1]), False)
        ]
        self.initialize()

    def __call__(self, X: jnp.ndarray):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def __str__(self) -> str:
        return super().__str__()
    
    def initialize(self) -> None:
        return super().initialize()
    
    def get_state(self) -> dict:
        return super().get_state()
    
    def apply_state(self, state: dict) -> None:
        return super().apply_state(state)