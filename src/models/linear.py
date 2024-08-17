import jax.numpy as jnp
import jax
import numpy as np
from typing import Any
from parents.module import Module
from parents.loss import Loss
from parents.optim import Optim
from layers.layers import LinearLayer

class LinearRegression(Module):

    def __init__(self, shape: tuple[int], bias: bool, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = LinearLayer(shape, bias)
        self.solver = "SGD"
        self.beta = 1e-1
        self.eta = 1e-2

    def __call__(self, X: jnp.ndarray, *args: Any, **kwds: Any) -> jnp.ndarray:
        return self.layer(X)
    
    def fit(self, optim: Optim, criterion: Loss, X: jnp.ndarray, Y: jnp.ndarray, *args: Any, **kwargs: Any) -> None:
        batch_size = max(int(X.shape[0] / 20), 5)
        last_update = 0
        last_err = 0
        err_bigger = False
        for epoch in range(100):
            if self.solver == "SGD":
                r_idx = np.random.randint(0, X.shape[0] - batch_size)
                r_X, r_Y = X[r_idx:r_idx + batch_size], Y[r_idx:r_idx + batch_size]
            else:
                r_X, r_Y = X, Y

            pred = self(r_X)
            err = criterion(pred, r_Y)
            d_err = criterion.backward(pred, r_Y)
            #d_pred = jnp.sum(self.layer.backward(r_X)[0], axis = 0)
            update = self.eta * jnp.sum(d_err, axis=0) + self.beta * last_update
            #update = self.eta * jnp.dot(d_pred.T, d_err) + self.beta * last_update
            self.layer.weights -= update
            if epoch > 2 and err > last_err:
                if err_bigger:
                    break
                err_bigger = True
            last_err = err
            last_update = update


    def initialize(self) -> None:
        self.layer.initialize()
    
    def __str__(self) -> str:
        return super().__str__()
    
    def get_state(self) -> dict:
        return super().get_state()
    
    def apply_state(self, state: dict) -> None:
        return super().apply_state(state)
    