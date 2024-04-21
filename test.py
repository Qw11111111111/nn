from layers.main import *
from loss_functions.main import *
from models.main import *
from optims.main import *
from utils.main import CV, train_model, cv_models, cv_optims
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()  
parser.add_argument("-l","--learning_rate", action="store", help="set the learning rate", default=1e-6, metavar="lr", type=float)
parser.add_argument("-a", "--alpha", action="store", help="the alpha paramter for momentum", default=1, metavar="alpha", type=float)
parser.add_argument("-b", "--batch_size", action="store", help="the batch size for SGD", default=10, metavar="B", type=int)
parser.add_argument("-e", "--epochs", action="store", help="the number of epochs", default=500, metavar="E", type=int)
args = parser.parse_args()

model_4 = DeepNet(fit_intercept=True, rng = 42)
model_4 = ShallowNet(fit_intercept=False, neurons=50, input_dim=1, rng = 42)
#model_4 = DeepNet(fit_intercept=False, input_dim=1)
#model_4 = Linear_regressor(input_dim=1, output_dim=1, rng=42)
#model_4 = VeryDeepModel(input_dim=1, rng=42, neurons=30, num_of_layers=5)
X_0 = np.array([np.linspace(i, 100, 10000) for i in range(13)]).T
X_0 = np.linspace(-100, 100, 100)
y_0 = np.sin(X_0)
y_0 = 1 / 200 * (X_0 ** 3 + 0.5 *  X_0 ** 2 - 2 * X_0) + 3 + np.random.normal(scale=100, size=len(y_0))
y_true = 2 * X_0
#y_true = 1 / 200 * (X_0 ** 3 + 0.5 *  X_0 ** 2 - 2 * X_0) + 3
y_0 = 2 * X_0 + np.random.normal(scale=50, size=len(y_0))
#y_0 = 4 * X_0 + 10
#y = 4 * X[:, 0] + 20 * X[:, 1] + np.sum([i * X[:, i] for i in range(2, 13)]) + 5 
plt.plot(X_0, y_0)
plt.plot(X_0, model_4.forward(X_0.reshape((X_0.shape[0], -1))))
plt.legend(["truth", "deep net"])
plt.grid(True)
plt.show()

EPOCHS = args.epochs
loss = MSELoss()
optimizer_deep = SGD(lr=args.learning_rate, model=model_4, loss=loss, batch_size=args.batch_size, momentum=True, alpha=args.alpha, normalization_rate=1, stop_val=1, kwargs={"dropout": False})
cv = CV(cv_folds=3)
indices = np.random.permutation(len(X_0))
X = np.array([X_0[i] for i in indices])
y = np.array([y_0[i] for i in indices])
TRAIN_SPLIT = int(len(X) * 0.8)
X_train, y_train = X[:TRAIN_SPLIT].reshape((X[:TRAIN_SPLIT].shape[0], -1)), y[:TRAIN_SPLIT].reshape((y[:TRAIN_SPLIT].shape[0], -1))
#print((z := model_4.get_state_dict()))

losses_4 = []
"""for train_indices, test_indices in cv.get_indices(X[:TRAIN_SPLIT]):
    print(train_indices, " | " , test_indices)
    for epoch in range(EPOCHS):
        optimizer_deep.backpropagation(X_train[train_indices], y_train[train_indices])
        losses_4.append(loss(Y=y[test_indices].reshape((y[test_indices].shape[0], -1)), pred=model_4.forward(X[test_indices].reshape((X[test_indices].shape[0], -1)))))
        if epoch % 100 == 0:
            print(f"epoch: {epoch}, test_loss: {losses_4[-1]}")
"""
#print(model_4.get_state_dict())

plt.plot(np.arange(len(losses_4)), losses_4)
plt.legend(["deep net"])
plt.grid(True)
plt.show()

model_4 = train_model(X.reshape(X.shape[0], -1), y.reshape(y.shape[0], -1), model_4, loss, optimizer_deep, EPOCHS, True, get_best=True)

plt.plot(X_0, y_true)
plt.plot(X_0, y_0)
plt.plot(X_0, model_4.forward(X_0.reshape((X_0.shape[0], -1))))
plt.legend(["truth", "data", "deep net"])
plt.grid(True)
plt.show()

model_4.load_state_dict(model_4.best_state_dict)
plt.plot(X_0, y_true)
plt.plot(X_0, y_0)
plt.plot(X_0, model_4.forward(X_0.reshape((X_0.shape[0], -1))))
plt.legend(["truth", "data", "deep net"])
plt.grid(True)
plt.show()