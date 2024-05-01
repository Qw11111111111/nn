import numpy as np
from parents.parentclasses import optim, Loss, Module

class SGD(optim):

    #TODO: optimize this process !!! (works, very slow for small batch size(naturally))

    def __init__(self, model: Module, loss: Loss, batch_size: int = 1, momentum: bool = False, alpha: float = 0.01, lr: float = 1e-4, stop_val: float = 1e-5, normalization_rate: float = 1, *args, **kwargs) -> None:
        super().__init__(lr, model, loss, stop_val, momentum, alpha, *args, normalization_rate, **kwargs)
        self.batch_size = batch_size
    
    def backpropagation(self, X: np.ndarray | float, Y: np.ndarray | float) -> None:
        random_perm = np.random.permutation(len(Y))
        X, Y = X[random_perm], Y[random_perm]
        for i in range(int(len(X) / self.batch_size) - 1):
            X_train, Y_train = X[i:i + self.batch_size], Y[i:i + self.batch_size]
            super().backpropagation(X_train, Y_train)


class Adam(optim):

    #TODO

    def __init__(self) -> None:
        super().__init__()

    def backpropagation(self) -> None:
        return super().backpropagation()
    
class Momentum(optim):
    
    #TODO: check if momentum is calculated correctly

    def __init__(self, lr: float, model: Module, loss: Loss, stop_val: float = 0.0001, beta: float = 5e-1, *args, **kwargs) -> None:
        super().__init__(lr, model, loss, stop_val, *args, **kwargs)
        assert beta < 1 and beta > 0, "beta must be in 1 > beta > 0"
        self.beta = beta
        self.prev_bias_grads = [0 for _ in range(len(self.model.layers))] if self.model.fit_intercept else None
        self.prev_weight_grads = [0 for _ in range(len(self.model.layers))]

    def backpropagation(self, X: np.ndarray, Y: np.ndarray) -> None:
        super().backpropagation(X, Y)
        for i, layer in enumerate(self.model.layers):
            if self.prev_bias_grads is not None:
                layer.bias_grad =  -self.beta * self.prev_bias_grads[i] + (1 - self.beta) * layer.bias_grad
                self.prev_bias_grads[i] = layer.bias_grad
            layer.weight_grad =  -self.beta * self.prev_weight_grads[i] + (1 - self.beta) * layer.weight_grad
            self.prev_weight_grads[i] = layer.weight_grad
            layer.update_state_dict(-self.lr * layer.weight_grad, -self.lr * layer.bias_grad)
    
class GD(optim):

    def __init__(self, lr: float, model: Module, loss: Loss, stop_val: float = 1e-4) -> None:
        """initalizes a basic gradient descent optimizer"""
        super().__init__(lr, model, loss, stop_val)

    def backpropagation(self, X: np.ndarray, Y: np.ndarray) -> None:
        super().backpropagation(X, Y)
        for layer in self.model.layers:
            layer.update_state_dict(-self.lr * layer.weight_grad, -self.lr * layer.bias_grad)


class AdaMax(optim):
    pass

class ElasticNet(optim):

    def __init__(self, lr: float, model: Module, loss: Loss, stop_val: float = 0.0001, *args, **kwargs) -> None:
        super().__init__(lr, model, loss, stop_val, *args, **kwargs)

    def backpropagation(self, X: np.ndarray, Y: np.ndarray) -> None:
        return super().backpropagation(X, Y)