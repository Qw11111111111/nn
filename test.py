from layers.main import *
from loss_functions.main import *
from models.main import *
from optims.main import *
import matplotlib.pyplot as plt

model_4 = DeepNet(fit_intercept=False)
model_4 = ShallowNet(fit_intercept=True, neurons=50, input_dim=1)
model_4 = DeepNet(fit_intercept=True)
print(model_4.forward(np.array([[1, 1, 2]]).T))
print(np.array([1, 1, 2]).reshape((1, np.array([1, 1, 2]).shape[0])))
X = np.linspace(0, 10, 1000)
y = np.sin(X)
y = 4 * X + 5 
plt.plot(X, y)
plt.plot(X, model_4.forward(X.reshape((X.shape[0], -1))))
plt.legend(["truth", "deep net"])
plt.grid(True)
plt.show()

EPOCHS = 5000
loss = MSELoss()
optimizer_deep = GD(lr=3e-7, model=model_4, loss=loss)
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

plt.plot(X, y)
plt.plot(X, model_4.forward(X.reshape((X.shape[0], -1))))
plt.legend(["truth", "deep net"])
plt.grid(True)
plt.show()