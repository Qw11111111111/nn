from layers.main import *
from loss_functions.main import *
from models.main import *
from optims.main import *
import matplotlib.pyplot as plt

model_4 = DeepNet(fit_intercept=False)

X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) 
plt.plot(X, y)
plt.plot(X, model_4.forward(X.reshape((X.shape[0], -1))))
plt.legend(["truth", "deep net"])
plt.grid(True)
plt.show()

EPOCHS = 200
loss = MSELoss()
optimizer_deep = GD(lr=3e-3, model=model_4, loss=loss)
TRAIN_SPLIT = int(len(X) * 0.8)
X_train, y_train = X[:TRAIN_SPLIT].reshape((X[:TRAIN_SPLIT].shape[0], -1)), y[:TRAIN_SPLIT].reshape((y[:TRAIN_SPLIT].shape[0], -1))
print((z := model_4.get_state_dict()))

losses_4 = []
for epoch in range(EPOCHS):
    for i in range(1, 80):
        optimizer_deep.backpropagation(X_train[i], y_train[i])
        print(model_4.get_state_dict())
    losses_4.append(loss.get_loss(Y=y[TRAIN_SPLIT:], pred=model_4.forward(X[TRAIN_SPLIT:].reshape((X[TRAIN_SPLIT:].shape[0], -1)))))

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