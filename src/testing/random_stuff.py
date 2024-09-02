from typing import Any
from src.layers import LinearLayer, ReLU
from src.parents import Module
import jax.numpy as jnp


class Model(Module):
    
    def __init__(self) -> None:
        self.layers = [
            LinearLayer((5, 2))
        ]
        
    def print(self):
        for row in self.layers[0].weights:
            print(*row)
    
    def __call__(self, X: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        return super().__call__(X, *args, **kwargs)

class Model2(Module):
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.layers = [
            LinearLayer((1, 1)),
            ReLU(),
            LinearLayer((1, 1))
        ]
        super().__init__(*args, **kwargs)
    
    def __call__(self, X: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        return super().__call__(X, *args, **kwargs)

model = Model()

class modelld(Model):
    pass

class Optimizer:
    
    def __init__(self, model: Model):
        self.model = model
    
    def step(self):
        self.model.layers[0].weights += 1


optimizer = Optimizer(model)

model.print()


for i in range(10):
    optimizer.step()
    
print("\npost\n")
model.print()


print("\n\n\n")

modle = modelld()
print(optimizer.__class__)

print(model)

print(modle)    

print(Model2())
    