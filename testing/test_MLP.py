from sklearn.datasets import make_blobs
from models.neural_nets import MLP
from neural_networks.loss import CrossEntropyLoss
from neural_networks.optims import Momentum
import numpy as np
import matplotlib.pyplot as plt
from utils.training import train_model

X, Y = make_blobs(1000, centers=2, random_state=42)
print(X[:10])
print(Y[:10])
print(X.shape)
print(Y.shape)
model = MLP(input_dim=X.shape[1], rng = 42)
loss = CrossEntropyLoss()
optim = Momentum(1e-7, model, loss)
EPOCHS = 500

colors = ["red", "blue"]
#print([colors[index] for index in np.argmax(model.forward(X), axis=1)])
plt.scatter(X.T[0], X.T[1], c=[colors[index] for index in np.argmax(model.forward(X), axis=1)])
plt.show()

model = train_model(X, Y, model, loss, optim, EPOCHS, True, get_best=True)
model.load_state_dict(model.best_state_dict)
plt.scatter(X.T[0], X.T[1], c=[colors[index] for index in np.argmax(model.forward(X), axis=1)])
plt.show()