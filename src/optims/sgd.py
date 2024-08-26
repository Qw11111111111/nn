from src.parents import Optim, Module
from typing import Any
import jax.numpy as jnp
import jax

class SGD(Optim):
    
    def __init__(self, model: Module, lr: float, batch_size: int = 10, *args: Any, **kwargs: Any) -> None:
        super().__init__(model, lr, *args, **kwargs)
        self.batch_size = batch_size
        
    def step(self, loss_grad: jnp.ndarray) -> None:
        return super().step(loss_grad)
    
    def set_params(self, *args: Any, **kwargs: Any) -> None:
        return super().set_params(*args, **kwargs)