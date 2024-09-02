from src.parents import Optim, Module, Loss
from typing import Any
import jax.numpy as jnp
import jax
import numpy as np

class SGD(Optim):
    
    def __init__(self, model: Module, criterion: Loss, lr: float = 1e-4, momentum: int = 1e-1, rng: int | None = None, batch_size: int = 1, *args: Any, **kwargs: Any) -> None:
        super().__init__(model, criterion, lr, momentum, *args, **kwargs)
        self.batch_size = batch_size
        
    def step(self, loss_grad: jnp.ndarray, X: jnp.ndarray) -> None:
        batch_start = self.rng.randint(0, X.shape[0] - self.batch_size) if self.rng is not None else np.random.randint(0, X.shape[0])
        batch = X[batch_start:batch_start + self.batch_size]
        return super().step(loss_grad, batch)
    
    def set_params(self, *args: Any, **kwargs: Any) -> None:
        return super().set_params(*args, **kwargs)