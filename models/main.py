import numpy as np
from loss_functions.main import *
from layers.main import *
from supers.main import *


class ShallowNet(Module):
    
    def __init__(self, neurons: int, input_dim: int, output_dim: int = 1, fit_intercept: bool = True, rng: int = None) -> None:
        self.neurons = neurons
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        layers = [LinearLayer(self.input_dim, self.neurons, fit_intercept=fit_intercept, rng=rng, position=0), ReLU(), LinearLayer(self.neurons, self.ouput_dim, fit_intercept=fit_intercept, rng=rng, position=1)]
        super().__init__(layers, rng, fit_intercept=fit_intercept)
    
    def forward(self, x: float | np.ndarray) -> np.ndarray:
        return super().forward(x)
    

