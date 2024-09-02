from src.optims import SGD
from src.loss import MSELoss
from src.models import MLP
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


X = jnp.arange(0., 100., 1000.).reshape((-1, 1))
Y = jnp.sqrt(X) + np.random.normal(0., 10., X.shape)

model = MLP(X.shape)
criterion = MSELoss()
optim = SGD(model, criterion, rng=42, batch_size=5)

print(optim.model)