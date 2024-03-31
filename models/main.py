import numpy as np
from loss_functions.main import *
from layers.main import *

class Module():

    def __init__(self, layers: list[Layer], rng: int = None) -> None:
        self.layers = layers
        if rng:
            self.rng = np.random.RandomState(rng)
        for layer in self.layers:
            layer.initialize()

    def forward(self, x: float | np.ndarray) -> float | np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def reset(self) -> None:
        for layer in self.layers:
            layer.initialize()

    def get_state_dict(self) -> dict:
        state_dict = {str(layer): {} for layer in self.layers}
        for layer in self.layers:
            if str(layer).endswith("no info"):
                continue
            state_dict.update({str(layer): layer.get_state()})
        
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        for layer in self.layers:
            if str(layer).endswith("no info"):
                continue
            layer.load_state(state_dict[str(layer)])

class ShallowNet(Module):
    
    def __init__(self, neurons: int, input_dim: int, output_dim: int = 1, fit_intercept: bool = True, rng: int = None) -> None:
        self.neurons = neurons
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        self.fit_intercept = fit_intercept
        layers = [LinearLayer(self.input_dim, self.neurons, fit_intercept=self.fit_intercept, rng=rng), ReLU(), LinearLayer(self.neurons, self.ouput_dim, fit_intercept=self.fit_intercept, rng=rng)]
        super().__init__(layers, rng)
    
    def forward(self, x: float | np.ndarray) -> np.ndarray:
        return super().forward(x)
    
        

class RegressionModel():
    """A Linear Regression model for multi dimensional linear regression tasks. Does not implement Ridge yet and has room to improve"""
    def __init__(self, lr: float, stop: float = 1e-4, input_size: int = 1, bias: bool = True, add_randomness: bool = False, momentum: bool = True, beta: int = 0.01, rng: np.random.RandomState = None, nonlinear: bool = False, lambda_val: float = 0, SGD: bool = False, batch_size: int = None) -> None:
        """A Linear Regression model for multi dimensional linear regression tasks. Takes the learning rate lr, the stop value of the backpropagation stop, the dimensionality of the input input_size,
         the beta value for momentum computation beta and booleans for momentum, bias and add_randomness, which adds randomness to the backpropagation. input_size is mandatory to set for inputs of dimension > 1"""
        self.add_bias = bias
        self.bias = 0
        self.lambda_value = lambda_val
        self.SGD = SGD
        if self.SGD:
            self.batch_size = batch_size
        if rng:
            self.weights = rng.random(input_size)
            if self.add_bias:
                self.bias = rng.random()
        else:
            self.weights = np.random.random(input_size)
            if self.add_bias:
                self.bias = np.random.random()
        self.lr = lr
        self.beta = beta
        self.input_size = input_size
        self.rng = rng
        if add_randomness:
            self.add_randomness = lambda x: x + (np.random.random() - 0.5) * self.lr 
        else:
            self.add_randomness = lambda x: x
        if momentum:
            self.add_momentum = lambda w, w_old: beta * (w - w_old)
        else:
            self.add_momentum = lambda *args: 0
        if nonlinear:
            self.x = lambda x: self.ReLU(x)
            self.dx = lambda x: self.d_ReLU(x)
        else:
            self.x = lambda x: x
            self.dx = lambda x: x
        self.prev_loss = 0
        self.stop = stop
        self.old_weights = self.weights
    
    def ReLU(self, X):
        return np.array([x if x > 0 else 0 for x in X])
    
    def d_ReLU(self, X):
        return np.array([1 if x > 0 else 0 for x in X])

    def backpropagation(self, X: np.ndarray, y: np.ndarray) -> None:
        """updates the weights and the bias of the model based on input prediction and label. 
        Loss is computed  using Mean Squared Error loss function within this method"""
        if self.SGD:
            idx = np.random.permutation(len(X))
            idx = np.split(idx, int(len(X) / self.batch_size))
        else:
            idx = np.arange(len(X))

        for index in idx:
            X_, y_= X[index], y[index]
            current_loss = loss(X_, self.weights, y_, self.lambda_value, self.dx)
            if abs(np.sum(current_loss) - np.sum(self.prev_loss)) > self.stop:
                v = - current_loss / np.sqrt(np.sum([x ** 2 for x in current_loss]))
                self.old_weights, self.weights = self.weights, self.weights + np.dot(self.lr, v) + self.add_randomness(0) + self.add_momentum(self.weights, self.old_weights)
                if self.add_bias:
                    self.bias = self.bias + self.lr * v 
                self.prev_loss = current_loss

    def forward(self, X) -> float:
        """predicts output of the model from input array x"""
        if np.isscalar(X):
            return self.weights[0] * self.x(X) + self.bias
        if len(X.shape) == 1:
            return  np.dot(self.x(X), self.weights) + self.bias
        return [np.sum(np.dot(self.x(x), self.weights)) + self.bias for x in X]
    
    def reset(self):
        if self.rng:
            self.weights = self.rng.random(self.input_size)
            if self.add_bias:
                self.bias = self.rng.random()
        else:
            self.weights = np.random.random(self.input_size)
            if self.add_bias:
                self.bias = np.random.random()

class shallow_net():
    
    def __init__(self, input_dim: int, neurons: int | np.ndarray, output_dim: int = 1,fit_intercept: bool = True, random_state: int | np.random.RandomState = None, activation_func = lambda x: np.array([[num if num > 0 else 0 for num in row] for row in x]), d_activation_func = lambda x: np.array([[1 if num > 0 else 0 for num in row] for row in x]), loss = MSELoss, lr: int = 1e-3, optim: str = "GD") -> None:
        """shallow neural network with variable number of neurons, variable input dimension and variable output dimension"""
        self.random_state = random_state
        self.neurons = neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fit_intercept = fit_intercept
        self.activation = activation_func
        self.d_activation = d_activation_func
        self.loss = loss
        self.lr = lr
        self.previous_losses: list[float] = [np.inf, np.inf]
        self.stop_val: float = 1
        self.optim = optim
        self.reset()

    def forward(self, x: np.ndarray | int) -> np.ndarray:
        """returns the prediction of the model on some input x"""
        if np.isscalar(x):
            x = np.array([[x]])
        if len(x.shape) == 1:
            x = x.reshape((-1, self.input_dim))
        return np.dot(self.activation(np.dot(x, self.weights) + self.biases), self.output_weights) + self.output_bias

    def backpropagation(self, x: np.ndarray, y: np.ndarray) -> None:
        """performs backpropagation on the model"""
        if np.isscalar(x):
            x = np.array([[x]])
        elif len(x.shape) == 1:
            x = x.reshape((-1, self.input_dim))
        if np.isscalar(y):
            x = np.array([[y]])
        elif len(y.shape) == 1:
            y = y.reshape((y.shape[0], 1))

        #forward pass
        f_0: np.ndarray = np.dot(x, self.weights) + self.biases
        h_0: np.ndarray = self.activation(f_0)
        f_1: np.ndarray = np.dot(h_0, self.output_weights) + self.output_bias

        if abs((self.previous_losses[-1] - (current := self.get_loss(x, y)))) < self.stop_val or current > self.previous_losses[1] and self.previous_losses[1] > self.previous_losses[0]:
            return
        
        # backward pass
        # iteratively compute the derivatives for each layer and each sample x
        if self.fit_intercept:
            del_bias: list = [[[], []] for _ in range(len(x))]
        prev = [[[], []] for _ in range(len(x))]
        del_weight = [[[], []] for _ in range(len(x))]
        
        # first attempt: "hard" code the derivatives and iterate over data instead of linalg
        for i, num in enumerate(y):
            prev[i][0] = self.loss(Y = num, pred = f_1[i], mode = "backprop")
            if self.fit_intercept:
                del_bias[i][0] = prev[i][0]
            del_weight[i][0] = prev[i][0] * h_0[i].T
            prev[i][1] = self.d_activation(self.output_weights.T * prev[i][0])
            if self.fit_intercept:
                del_bias[i][1] = prev[i][1]
            del_weight[i][1] = prev[i][1] * num.T if not np.isscalar(num) else prev[i][1] * num
        
        # sum up all derivatives over the corresponding axes to get the gradients
        direction = lambda x: np.array(- x / np.sqrt(np.sum([num ** 2 for num in x]))).reshape((len(x), -1)) if not isinstance(x, float) else - x

        output_weight_grad = direction(sum([num[0] for num in del_weight]) / len(del_weight))
        hidden_weight_grad = direction(sum([num[1] for num in del_weight]) / len(del_weight))
        if self.fit_intercept:
            output_bias_grad = direction(sum([num[0] for num in del_bias]) / len(del_bias)) 
            hidden_bias_grad = direction(sum([num[1] for num in del_bias]) / len(del_bias))
    	    
        # update params
        self.output_weights += self.lr * output_weight_grad
        self.weights += self.lr * hidden_weight_grad
        if self.fit_intercept:
            self.output_bias += self.lr * output_bias_grad
            self.biases += self.lr * hidden_bias_grad
        self.previous_losses[0], self.previous_losses[1] = self.previous_losses[1], current

        # second attempt: use a function to calculate the derivatives

        """for i, num in enumerate(x):
            for layer in range(2):
                prev[layer, i] = self.loss(Y = num, pred = f_1[i])
                del_weight[layer, i] = ... * prev[layer, i]
                if self.fit_intercept:
                    del_bias[layer, i] = ...
        def backward(self, iteration: int = None, prev_d: int = None, del_weights: np.ndarray = None, del_biases: np.ndarray = None, X: np.ndarray = None, Y: np.ndarray = None):

            pass"""

    def reset(self, input_dim: int = None, output_dim: int = None, neurons: int | np.ndarray = None, fit_intercept: bool = None, random_state: int | np.random.RandomState = None, activation_func = None, d_activation_func = None, loss = None, lr: int = None, optim: str = None) -> None:
        """Sets the params of the model. All inputs are optional and are only used to change he model params"""
        
        if input_dim:
            self.input_dim = input_dim
        if output_dim:
            self.output_dim = output_dim
        if neurons:
            self.neurons = neurons
        if fit_intercept:
            self.fit_intercept = fit_intercept
        if random_state:
            self.random_state = random_state
        if activation_func:
            self.activation = activation_func
        if d_activation_func:
            self.d_activation = d_activation_func
        if loss:
            self.loss = loss
        if lr:
            self.lr = lr
        if optim:
            self.optim = optim
        self.previous_loss = [np.inf, np.inf]
        
        if not self.random_state:
            self.output_weights: np.ndarray = np.random.random((self.neurons, self.output_dim))
            self.weights: np.ndarray = np.random.random((self.input_dim, self.neurons))
            self.biases: np.ndarray = np.zeros((1, self.neurons))
            self.output_bias: np.ndarray = np.zeros((1, self.output_dim))
            if self.fit_intercept:
                self.output_bias: np.ndarray = np.random.random((1, self.output_dim))
                self.biases: np.ndarray = np.random.random((1, self.neurons))
        
        elif isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
            self.output_weights = self.random_state.random((self.neurons, self.output_dim))
            self.weights: np.ndarray = self.random_state.random((self.input_dim, self.neurons))
            self.biases = np.zeros((1, self.neurons))
            self.output_bias = np.zeros((1, self.output_dim))
            if self.fit_intercept:
                self.output_bias: np.ndarray = self.random_state.random((1, self.output_dim))
                self.biases = self.random_state.random((1, self.neurons))
        
        else:
            self.output_weights = self.random_state.random((self.neurons, self.output_dim))
            self.weights: np.ndarray = self.random_state.random((self.input_dim, self.neurons))
            self.biases = np.zeros((1, self.neurons))
            if self.fit_intercept:
                self.output_bias: np.ndarray = self.random_state.random((1, self.output_dim))
                self.biases = self.random_state.random((1, self.neurons))
    
    def get_loss(self, X: np.ndarray, y_true: np.ndarray):
        """returns the loss"""
        pred = self.forward(X)
        return self.loss(pred=pred, Y=y_true)