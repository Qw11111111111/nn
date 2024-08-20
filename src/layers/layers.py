from typing import Any
import jax.numpy as jnp
from jax.numpy import ndarray
import jax.random as jrd
import jax
from numpy import random
from parents.layer import Layer

class LinearLayer(Layer):

    def __init__(self, shape: tuple[int, int], bias: bool = True, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fit_bias = bias
        self.shape = shape
        self.initialize()

    def __call__(self, X: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        return jnp.dot(X, self.weights) #+ (self.bias if self.fit_bias else 0)
    
    def __str__(self) -> str:
        return super().__str__()
    
    def backward(self, X: jnp.ndarray) -> jnp.ndarray:
        return jax.jacobian(self)(X)
    
    def initialize(self):
        self.weights = random.normal(0., 2 / self.shape[1], self.shape)
        self.bias = random.normal(0., 2 / self.shape[1], self.shape[1])

class ReLU(Layer):
    #todo
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, X: jnp.ndarray, *args: Any, **kwds: Any) -> Any:
        return X
    
    def __str__(self) -> str:
        return super().__str__()
    
    def backward(self, X: ndarray) -> ndarray:
        return X
    
    def initialize(self):
        return super().initialize()