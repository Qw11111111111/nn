from parents.layer import Layer
from layers.layers import LinearLayer, ReLU
from parents.module import Module
from parents.loss import Loss
from parents.optim import Optim
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

    def __call__(self, X: jnp.ndarray):
        for layer in self.layers:
            X = layer(X)
        return X

    def fit(self, optim: Optim, criterion: Loss, X: jnp.ndarray, Y: jnp.ndarray, *args: Any, **kwargs: Any) -> None:
        return super().fit(optim, criterion, X, Y, *args, **kwargs)
    
    def __str__(self) -> str:
        return super().__str__()
    
    def initialize(self) -> None:
        return super().initialize()
    
    def get_state(self) -> dict:
        return super().get_state()
    
    def apply_state(self, state: dict) -> None:
        return super().apply_state(state)