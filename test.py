from layers.main import *
from loss_functions.main import *
from models.main import *
from optims.main import *
import matplotlib.pyplot as plt

model_4 = DeepNet(fit_intercept=False)
model_4 = ShallowNet(fit_intercept=True, neurons=50, input_dim=13)
#model_4 = DeepNet(fit_intercept=False, input_dim=2)
#model_4 = Linear_regressor(input_dim=2, output_dim=1)
#model_4 = VeryDeepModel(input_dim=2)
X = np.array([np.linspace(i, 100, 1000) for i in range(13)]).T
y = np.sum(np.sin(X))
y = 4 * X[:, 0] + 20 * X[:, 1] + np.sum([i * X[:, i] for i in range(2, 13)]) + 5 
plt.plot(X[:, 0], y)
plt.plot(X[:, 0], model_4.forward(X))
plt.legend(["truth", "deep net"])
plt.grid(True)
plt.show()

EPOCHS = 5000
loss = MSELoss()
optimizer_deep = GD(lr=1e-5, model=model_4, loss=loss)
TRAIN_SPLIT = int(len(X) * 0.8)
X_train, y_train = X[:TRAIN_SPLIT].reshape((X[:TRAIN_SPLIT].shape[0], -1)), y[:TRAIN_SPLIT].reshape((y[:TRAIN_SPLIT].shape[0], -1))
print((z := model_4.get_state_dict()))

losses_4 = []
for epoch in range(EPOCHS):
    
    optimizer_deep.backpropagation(X_train, y_train)
    losses_4.append(loss(Y=y[TRAIN_SPLIT:].reshape((y[TRAIN_SPLIT:].shape[0], -1)), pred=model_4.forward(X[TRAIN_SPLIT:].reshape((X[TRAIN_SPLIT:].shape[0], -1)))))

print(z)

plt.plot(np.arange(len(losses_4)), losses_4)
plt.legend(["deep net"])
plt.grid(True)
plt.show()

plt.plot(X[:, 0], y)
plt.plot(X[:,0], model_4.forward(X.reshape((X.shape[0], -1))))
plt.legend(["truth", "deep net"])
plt.grid(True)
plt.show()