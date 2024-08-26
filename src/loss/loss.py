import jax.numpy as jnp
import jax
from typing import Any
from src.parents import Loss

class MSELoss(Loss):
    
    def __init__(self, *args:Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def backward(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(self)(X, Y)
    
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray, *args: Any, **kwds: Any) -> jnp.ndarray:
        return 1 / (1 * X.shape[0]) * jnp.sum(jnp.square(Y - X))