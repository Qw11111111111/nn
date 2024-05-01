import numpy as np
from neural_networks.layers import linear
from parents.parentclasses import Module

class Linear_regressor(Module):

    def __init__(self, rng: int = None, fit_intercept: bool = True, input_dim: int = 1, output_dim: int = 1) -> None:
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        layers = [
            linear.LinearLayer(input_dim, output_dim, fit_intercept, rng)
        ]
        super().__init__(layers, rng, fit_intercept)
