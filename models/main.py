from re import M
import numpy as np
from loss_functions.main import *
from layers.main import *
from supers.main import *
from supers.main import Layer


class ShallowNet(Module):
    
    def __init__(self, neurons: int, input_dim: int, output_dim: int = 1, fit_intercept: bool = True, rng: int = None) -> None:
        self.neurons = neurons
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        layers = [LinearLayer(self.input_dim, self.neurons, fit_intercept=fit_intercept, rng=rng, position=0), ReLU(position=1), LinearLayer(self.neurons, self.ouput_dim, fit_intercept=fit_intercept, rng=rng, position=2)]
        super().__init__(layers, rng, fit_intercept=fit_intercept)
    
    def forward(self, x: float | np.ndarray) -> np.ndarray:
        return super().forward(x)
    

class DeepNet(Module):

    def __init__(self, rng: int = None, fit_intercept: bool = True, input_dim: int = 1, output_dim: int = 1) -> None:
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        layers = [
            LinearLayer(input_dim, 10, 10, fit_intercept, position=0),
            ReLU(position=1),
            LinearLayer(10, 5, 5, fit_intercept, position=2),
            ReLU(position=3),
            LinearLayer(5, 1, 1, fit_intercept, position=4)
        ]

        super().__init__(layers, rng, fit_intercept)
    
class Linear_regressor(Module):

    def __init__(self, rng: int = None, fit_intercept: bool = True, input_dim: int = 1, output_dim: int = 1) -> None:
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        layers = [
            LinearLayer(input_dim, output_dim, fit_intercept, rng)
        ]
        super().__init__(layers, rng, fit_intercept)

