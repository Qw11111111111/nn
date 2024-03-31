import numpy as np

class Layer():

    def __init__(self, rng: int = None) -> None:
        if rng:
            self.rng = np.random.RandomState(rng)
        else:
            self.rng = None
        pass

    def get_grad(self, prev: np.ndarray | float, X: np.ndarray | float) -> list[np.ndarray]:
        pass

    def forward(self, *args):
        return args

    def initialize(self):
        pass

    def get_state(self):
        pass
    
    def load_state(self):
        pass

    def update_state_dict(self, weight_update: list, bias_upsate: list = None) -> None:
        pass

    def __str__(self) -> str:
        pass

class Module():

    def __init__(self, layers: list[Layer], rng: int = None, fit_intercept: bool = True) -> None:
        self.layers = layers
        if rng:
            self.rng = np.random.RandomState(rng)
        for layer in self.layers:
            layer.initialize()
        self.fit_intercept = fit_intercept

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
        
    def backprop_forward(self, X: np.ndarray | float) -> list[np.ndarray | float, str]:
        Y = [X]
        Names = []
        for layer in self.layers:
            Y.append(layer.forward(Y[-1]))
            Names.append(str(layer))
        return Y, Names
    
    def get_grad(self, prev_grad: np.ndarray | float, layer_idx: int, input: np.ndarray, input_to_activ: np.ndarray = None, mode: str = "hidden") -> list[np.ndarray]:
        if mode == "hidden":
            if not str(self.layers[layer_idx + 1]).startswith("activation layer"):
                activation = np.diag(np.ones(input_to_activ.shape[0], dtype=float))
            else:
                activation = self.layers[layer_idx - 1].get_grad(prev_grad, X=input_to_activ)
            print(self.layers[layer_idx], self.layers[layer_idx - 1])
            return *self.layers[layer_idx].get_grad(prev_grad, input), activation
        return self.layers[layer_idx].get_grad(prev_grad, input)
    
    def apply_grad(self, weight_grad: dict, bias_grad: dict) -> None:
        for i, layer in enumerate(self.layers):
            """print(str(layer), "huhuhuhuuh")
            func = weight_grad[str(layer)]
            print(func)
            print(func(np.ones((10, 1))).shape, "wtf")"""
            layer.update_state_dict(weight_grad[str(layer)], bias_grad[str(layer)] if self.fit_intercept else None)
        ...

class Loss():

    def __init__(self) -> None:
        pass

    def get_loss(self) -> np.ndarray | float:
        pass
    
    def  get_grad(self)-> np.ndarray:
        pass

class optim():
    
    def __init__(self, lr: float, model: Module, loss: Loss, stop_val: float = 1e-4) -> None:
        self.lr = lr
        self.model = model
        self.stop_val = stop_val
        self.prev_losses = [np.inf, np.inf]
        self.loss_model = loss

    def backpropagation(self, X: np.ndarray | float, Y: np.ndarray | float) -> None:

        if np.isscalar(X):
            X = np.array([[X]])
        elif len(X.shape) == 1:
            X = X.reshape((-1, X.shape[0]))
        if np.isscalar(Y):
            X = np.array([[Y]])
        elif len(Y.shape) == 1:
            Y = Y.reshape((Y.shape[0], 1))

        FORWARD, NAMES = self.model.backprop_forward(X)
        print(NAMES)
        loss = self.loss_model.get_loss(pred=FORWARD[-1], Y=Y)
        
        if abs((self.prev_losses[-1] - loss)) < self.stop_val or loss > self.prev_losses[1] and self.prev_losses[1] > self.prev_losses[0]:
            return
        
        model_weight_grads = {name: [] for name in NAMES}
        if self.model.fit_intercept:
            model_bias_grads = {name: [] for name in NAMES}
        
        prev_grad = self.loss_model.get_grad(pred=FORWARD[-1], Y=Y)
        
        for i, name in enumerate(reversed(NAMES[2:])):
            i += 1
            print(prev_grad, "prevgrad")
            print(i, "i")
            weight_grad, bias_grad, weights, activation_grad = self.model.get_grad(prev_grad, -i, FORWARD[-i - 1], FORWARD[-i - 2])
            if self.model.fit_intercept:
                model_bias_grads[name] = bias_grad
            model_weight_grads[name] = weight_grad
            print(activation_grad.shape, weights.T.shape, prev_grad.shape)
            prev_grad = np.dot(np.dot(activation_grad, weights.T).T, prev_grad)
            print(prev_grad, "after")
        return_val = self.model.get_grad(prev_grad.reshape((-1, prev_grad.shape[0])), 0, X, mode="input")
        print(return_val)
        weight_grad, bias_grad, weights = return_val[0], return_val[1], return_val[2]
        model_weight_grads[NAMES[0]] = weight_grad
        if self.model.fit_intercept:
            model_bias_grads[NAMES[0]] = bias_grad

        get_direction = lambda x: np.array(- x / np.sqrt(np.sum([[num ** 2 for num in row] for row  in x]))).reshape((len(x), -1)) if not isinstance(x, float) else - x

        for i, name in enumerate(list(model_weight_grads.keys())):
            grad = model_weight_grads[name]
            if not isinstance(grad, np.ndarray):
                continue
            print(grad, "grad")
            print(grad.shape, "GRAD SHAPE")
            print(self.lr * get_direction(grad), "hallo")
            print(np.ones((grad.shape)) + get_direction(grad))
            print(name)
            print(model_weight_grads)
            model_weight_grads[name] = lambda x: x + self.lr * get_direction(grad).reshape(x.shape)
            print(model_weight_grads, "done")
        
        if self.model.fit_intercept:
            for i, name in enumerate(list(model_bias_grads.keys())):
                grad = model_bias_grads[name]
                if not grad:
                    continue
                model_bias_grads[name] = lambda x: x + self.lr * get_direction(grad)
            
        self.model.apply_grad(model_weight_grads, model_bias_grads if self.model.fit_intercept else None)

        self.prev_losses[0], self.prev_losses[1] = self.prev_losses[1], loss

