import numpy as np
from neural_networks.loss import *
from neural_networks.layers import activations, linear, conv
from parents.parentclasses import Module

class ShallowNet(Module):
    
    def __init__(self, neurons: int, input_dim: int, output_dim: int = 1, fit_intercept: bool = True, rng: int = None) -> None:
        self.neurons = neurons
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        layers = [linear.LinearLayer(self.input_dim, self.neurons, fit_intercept=fit_intercept, rng=rng, position=0), activations.ReLU(position=1), linear.LinearLayer(self.neurons, self.ouput_dim, fit_intercept=fit_intercept, rng=rng, position=2)]
        super().__init__(layers, rng, fit_intercept=fit_intercept)
    
    def forward(self, x: float | np.ndarray) -> np.ndarray:
        return super().forward(x)
    

class DeepNet(Module):

    def __init__(self, rng: int = None, fit_intercept: bool = True, input_dim: int = 1, output_dim: int = 1) -> None:
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        layers = [
            linear.LinearLayer(input_dim, 10, fit_intercept, position=0),
            activations.ReLU(position=1),
            linear.LinearLayer(10, 5, fit_intercept, position=2),
            activations.ReLU(position=3),
            linear.LinearLayer(5, 1, fit_intercept, position=4)
        ]

        super().__init__(layers, rng, fit_intercept)
    

class VeryDeepModel(Module):

    def __init__(self, rng: int = None, fit_intercept: bool = True, input_dim: int = 1, output_dim: int = 1, num_of_layers: int = 5, neurons: int = 20) -> None:
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        layers = [
            linear.LinearLayer(input_dim, neurons, fit_intercept, position=0), 
            activations.ReLU(position=0) 
        ]
        for i in range(1, num_of_layers):
            layers += [linear.LinearLayer(neurons, neurons, fit_intercept, rng, i), activations.ReLU(i)]
        layers += [linear.LinearLayer(neurons, output_dim, fit_intercept, rng, i + 1)]

        super().__init__(layers, rng, fit_intercept)



