import numpy as np
import matplotlib.pyplot as plt

class Module:
    pass

class optim:
    pass

class Loss:
    pass

def MSELoss(X: np.ndarray | int = None, Y: np.ndarray | int = None, w: np.ndarray | int = None, pred: np.ndarray | int = None, mode: str = "forward") -> float | np.ndarray:
    if X and w:
        pred = np.dot(X, w)
    if mode == "forward":
        return np.sum([(pred[i] - Y[i]) ** 2 for i in range(len(Y))]) / len(Y)
    else:
        return 2 * (pred[0] - Y[0])

def loss(X, w, Y, lambda_value: float = 0, func = lambda x: x) ->  float:
    """MSE Loss function, takes a np.array or float PRED and a np.array or float Y and returns the gradient of the MSE loss"""
    if np.isscalar(X):
        X = np.array([X])
    if np.isscalar(Y):
        Y = np.array([Y])
    return - np.dot(np.transpose(X), (Y - np.dot(X, func(w)))) + 2 * lambda_value * func(w)

def argwhere(v: list | np.ndarray, val: str | float) -> int:
    for i, value in enumerate(v):
        if str(value) == str(val):
            return i

class CV():

    #TODO: fix get_indices

    def __init__(self, cv_folds: int = 5, permute: bool = True):
        self.cv_folds = cv_folds
        self.permute = permute

    def get_indices(self, X: np.ndarray):
        if self.permute:
            permutation = np.random.permutation(len(X))
        else:
            permutation = np.arange(len(X))
        fold_size = int(len(permutation) / self.cv_folds)
        indices = [permutation[i * fold_size:i * fold_size + fold_size] for i in range(self.cv_folds)]
        print(indices)
        for i in range(self.cv_folds):
            yield indices[:i] + indices[i + 1:], indices[i]

def cv_models(X: np.ndarray, Y: np.ndarray, models: list[Module], loss: Loss, optim: optim, splits: int = 10, epochs: int = 5000, verbose: bool = False, permute: bool = True) -> Module:
    kfold = CV(splits, permute)
    test_losses = np.empty(len(models))
    for train_indices, test_indices in kfold.get_indices(X):
        for i, model in enumerate(models):
            optim.set_params(model = model)
            train_losses = []
            for epoch in range(epochs):
                optim.backpropagation(X[train_indices], Y[train_indices])
                if verbose:
                    train_losses.append(loss(Y=Y[test_indices].reshape((Y[train_indices].shape[0], -1)), pred=model.forward(X[train_indices].reshape((X[train_indices].shape[0], -1)))))
                if epoch % 100 == 0 and verbose:
                    print(f"epoch: {epoch}, test_loss: {train_losses[-1]}")
            test_losses[i] += loss(Y=Y[test_indices].reshape((Y[test_indices].shape[0], -1)), pred=model.forward(X[test_indices].reshape((X[test_indices].shape[0], -1)))) / splits
        
    model = models[np.argmin(test_losses)]
    return model

def cv_optims(X: np.ndarray, Y: np.ndarray, model: Module, loss: Loss, optims: list[optim], splits: int = 10, epochs: int = 5000, verbose: bool = False, permute: bool = True) -> optim:
    kfold = CV(splits, permute)
    test_losses = np.empty(len(optims))
    for train_indices, test_indices in kfold.get_indices(X):
        for i, optim in enumerate(optims):
            train_losses = []
            for epoch in range(epochs):
                optim.backpropagation(X[train_indices], Y[train_indices])
                if verbose:
                    train_losses.append(loss(Y=Y[test_indices].reshape((Y[train_indices].shape[0], -1)), pred=model.forward(X[train_indices].reshape((X[train_indices].shape[0], -1)))))
                if epoch % 100 == 0 and verbose:
                    print(f"epoch: {epoch}, test_loss: {train_losses[-1]}")
            test_losses[i] += loss(Y=Y[test_indices].reshape((Y[test_indices].shape[0], -1)), pred=model.forward(X[test_indices].reshape((X[test_indices].shape[0], -1)))) / splits
        
    optim = optims[np.argmin(test_losses)]
    return optim

def train_model(X: np.ndarray, Y: np.ndarray, model: Module, loss: Loss, optim: optim, epochs: int = 5000, verbose: bool = False, train_split: float = 0.8, get_best: bool = False) -> Module:
    train_losses = []
    test_losses = []
    min = np.inf
    TRAIN_SPLIT = int(len(X) * train_split)
    for epoch in range(epochs):
        optim.backpropagation(X[:TRAIN_SPLIT], Y[:TRAIN_SPLIT])
        if get_best:
            test_loss = loss(Y=Y[TRAIN_SPLIT:].reshape((Y[TRAIN_SPLIT:].shape[0], -1)), pred=model.forward(X[TRAIN_SPLIT:].reshape((X[TRAIN_SPLIT:].shape[0], -1))))
            if test_loss < min:
                model.best_state_dict = model.get_state_dict()
                min = test_loss
        if verbose:
            train_losses.append(loss(Y=Y[:TRAIN_SPLIT].reshape((Y[:TRAIN_SPLIT].shape[0], -1)), pred=model.forward(X[:TRAIN_SPLIT].reshape((X[:TRAIN_SPLIT].shape[0], -1)))))
            test_losses.append(loss(Y=Y[TRAIN_SPLIT:].reshape((Y[TRAIN_SPLIT:].shape[0], -1)), pred=model.forward(X[TRAIN_SPLIT:].reshape((X[TRAIN_SPLIT:].shape[0], -1)))))
        if epoch % 100 == 0 and verbose:
            print(f"{f"epoch: {epoch}":<15} | {f"train_loss: {train_losses[-1]:.3g}":^25} | {f"test_loss: {test_losses[-1]:.3g}":^25}")

    if verbose:
        plt.plot(np.arange(epochs), train_losses)
        plt.plot(np.arange(epochs), test_losses)
        plt.legend(["train losses", "test losses"])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(True)
        plt.show()

    return model

def l2(w: np.ndarray, y: np.ndarray = None, grad: bool = False) -> float:
    if grad:
        # assuming l2 ** 2
        if not y:
            return 2 * np.sum(w)
        else:
            return 2 * np.sum(w + y)
    if not y:
        return np.sqrt(np.sum(np.square(w)))
    else:
        return np.sqrt(np.sum(np.square(w + y)))