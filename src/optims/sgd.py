from src.parents import Optim
from typing import Any
import jax.numpy as jnp
import jax

class SGD(Optim):
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
    def backward(self) -> Any | None:
        return super().backward()
    
    def set_params(self, *args: Any, **kwargs: Any) -> None:
        return super().set_params(*args, **kwargs)