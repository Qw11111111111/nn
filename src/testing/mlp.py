import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from loss.loss import MSELoss
from models.neural_net import MLP

global VERBOSE, rng, X, Y

VERBOSE = True
rng = np.random.Generator(np.random.PCG64())
BIAS = False
EPOCHS = 100

true_w = 2
true_b = 0

X = jnp.arange(0., 500., 0.2).reshape((-1, 1))
Y = X * true_w + true_b + rng.normal(0., 10., X.shape)

TRAIN_SPLIT = int(0.8 * X.shape[0])

my_model = MLP((X.shape[1], Y.shape[1]), BIAS)

my_optim = None
my_loss = MSELoss()

def plot():
    plt.scatter(X, Y, s=0.5)
    plt.plot(X, my_model(X))
    plt.legend(["data", "mine"])
    plt.show()

if VERBOSE:
    plot()

my_model.fit(my_optim, my_loss, X[:TRAIN_SPLIT], Y[:TRAIN_SPLIT], EPOCHS)

plot()
