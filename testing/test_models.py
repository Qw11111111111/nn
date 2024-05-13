from neural_networks.loss import MSELoss
from models.linear_regression import Linear_regressor
from models.neural_nets import *
from neural_networks.optims import GD, SGD, Momentum, ElasticNet
from utils.training import CV, train_model, cv_models, cv_optims
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser()  
parser.add_argument("-l","--learning_rate", action="store", help="set the learning rate", default=1e-9, metavar="lr", type=float)
parser.add_argument("-a", "--alpha", action="store", help="the alpha paramter for momentum", default=1, metavar="alpha", type=float)
parser.add_argument("-b", "--batch_size", action="store", help="the batch size for SGD", default=10, metavar="B", type=int)
parser.add_argument("-e", "--epochs", action="store", help="the number of epochs", default=500, metavar="E", type=int)
args = parser.parse_args()

torch_model = nn.Sequential(
    nn.Linear(1, 500),
    nn.ReLU(),
    nn.Linear(500, 1)
)

model_4 = DeepNet(fit_intercept=True, rng = 42)
model_4 = ShallowNet(fit_intercept=True, neurons=50, input_dim=1, rng = 42)
#model_4 = DeepNet(fit_intercept=False, input_dim=1)
#model_4 = Linear_regressor(input_dim=1, output_dim=1, rng=42, fit_intercept=False)
#model_4 = VeryDeepModel(input_dim=1, rng=42, neurons=30, num_of_layers=5)
#X_0 = np.array([np.linspace(i, 100, 10000) for i in range(13)])
X_0 = np.linspace(-100, 100, 1000)
#y_0 = np.sin(X_0)
y_0 = 1 / 50 * np.square(X_0) + 10 * X_0 + np.random.normal(scale=10, size=len(X_0))
y_true = 1 / 50 * np.square(X_0) + 10 * X_0
X_0, y_0, y_true = X_0.reshape((X_0.shape[0], 1)), y_0.reshape((y_0.shape[0], 1)), y_true.reshape((y_true.shape[0], 1))
print(X_0.shape, y_0.shape)
#y_true = 1 / 200 * (X_0 ** 3 + 0.5 *  X_0 ** 2 - 2 * X_0) + 3
#y_0 = 2 * X_0 + np.random.normal(scale=50, size=len(y_0))
#y_0 = 4 * X_0 + 10
#y = 4 * X[:, 0] + 20 * X[:, 1] + np.sum([i * X[:, i] for i in range(2, 13)]) + 5 
plt.plot(X_0, y_0)
plt.plot(X_0, y_true)
plt.plot(X_0, model_4.forward(X_0))
plt.plot(X_0, torch_model.forward(torch.from_numpy(X_0)).detach().numpy())
plt.legend(["data", "truth", "deep net", "torch"])
plt.grid(True)
plt.show()

EPOCHS = args.epochs
loss_ = MSELoss()
optimizer = torch.optim.SGD(torch_model.parameters(), lr = args.learning_rate, momentum=args.alpha)
loss_func = torch.nn.MSELoss()
optimizer_deep = SGD(lr=args.learning_rate, model=model_4, loss=loss_, batch_size=args.batch_size, momentum=True, alpha=args.alpha, normalization_rate=1, stop_val=1, kwargs={"dropout": False})
optimizer_deep = Momentum(lr=args.learning_rate, model=model_4, loss=loss_, batch_size=args.batch_size, momentum=True, beta=args.alpha, normalization_rate=1, stop_val=1, kwargs={"dropout": False})
optimizer_deep = ElasticNet(lr=args.learning_rate, model=model_4, loss=loss_, batch_size=args.batch_size, momentum=True, normalization_rate=1, stop_val=1, kwargs={"dropout": False})
losses_4 = []

for epoch in range(EPOCHS):
    
    optimizer_deep.backpropagation(X_0, y_0)
    losses_4.append(loss_(model_4.forward(X_0), y_0))
    pred = torch_model.forward(torch.from_numpy(X_0))
    criterion= loss_func(pred, torch.from_numpy(y_0))
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()


plt.plot(np.arange(len(losses_4)), losses_4)
plt.legend(["deep net"])
plt.grid(True)
plt.show()

model_4.reset()

model_4 = train_model(X_0, y_0, model_4, loss_, optimizer_deep, EPOCHS, True, get_best=True)


plt.plot(X_0, y_true)
plt.plot(X_0, y_0)
plt.plot(X_0, model_4.forward(X_0))
plt.plot(X_0, torch_model.forward(torch.from_numpy(X_0)).detach().numpy())
plt.legend(["truth", "data", "deep net", "torch"])
plt.grid(True)
plt.show()
#print(torch_model.forward(torch.from_numpy(X_0).unsqueeze(1)).detach().squeeze().numpy())
model_4.load_state_dict(model_4.best_state_dict)
plt.plot(X_0, y_true)
plt.plot(X_0, y_0)
plt.plot(X_0, model_4.forward(X_0))
plt.plot(X_0, torch_model.forward(torch.from_numpy(X_0)).detach().numpy())
plt.legend(["truth", "data", "deep net", "torch"])
plt.grid(True)
plt.show()