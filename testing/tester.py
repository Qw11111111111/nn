from template import shallow_net
from neural_networks.loss import *
from models.neural_nets import *
from neural_networks.optims import *
import matplotlib.pyplot as plt
import torch
from torch import nn
torch.set_default_dtype(torch.float64)





# the onliner:
"""model = nn.Sequential(nn.Linear(1,1,False), nn.ReLU(), nn.Linear(1,1,False))
crit = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), 1e-5)
X = 100
#done
for ep in 10:
    pred = model(X)
    loss = crit(pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()"""

(train := lambda model, crit, optimizer, X, y: [(zero := lambda optimizer: optimizer.zero_grad())(optimizer)(get_loss := lambda model, X: crit(model(X), y).backward())(model, X, y)(step := lambda optimizer: optimizer.step())(optimizer) for ep in range(10)])(model := nn.Sequential(nn.Linear(1,1,False), nn.ReLU(), nn.Linear(1,1,False)), nn.MSELoss(), torch.optim.SGD(model.parameters(), 1e-5), torch.tensor([100, 200, 300]), torch.tensor([1, 2, 3]))








class Linear_Regression_model(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=20, bias=False),
            nn.ReLU(),
            nn.Linear(20, 1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

model = ShallowNet(30, 1, fit_intercept=False)
model_2 = shallow_net(1, 100, fit_intercept=False, lr=0.07)
model_3 = Linear_Regression_model()
model_4 = DeepNet(fit_intercept=False)

X = np.linspace(0, 4 * np.pi, 1000)
y = np.sin(X) 
plt.plot(X, y)
plt.plot(X, model.forward(X.reshape((X.shape[0], -1))))
plt.plot(X, model_2.forward(X.reshape((X.shape[0], -1))))
plt.plot(X, model_3.forward(torch.from_numpy(X.reshape((X.shape[0], -1)))).detach().numpy())
plt.plot(X, model_4.forward(X.reshape((X.shape[0], -1))))
plt.legend(["truth", "shallow net", "model2", "torch", "deep net"])
plt.grid(True)
plt.show()

EPOCHS = 2000
criterion =  nn.MSELoss()
optimizer_2 = torch.optim.Adam(params=model_3.parameters(), lr=1e-7)
loss = MSELoss()
optimizer = GD(lr=3e-3, model = model, loss = loss)
optimizer_deep = GD(lr=3e-6, model=model_4, loss=loss)
TRAIN_SPLIT = int(len(X) * 0.8)
X_train, y_train = X[:TRAIN_SPLIT].reshape((X[:TRAIN_SPLIT].shape[0], -1)), y[:TRAIN_SPLIT].reshape((y[:TRAIN_SPLIT].shape[0], -1))
print((z := model.get_state_dict()))
losses = []
losses_2 = []
losses_3 = []
losses_4 = []
for epoch in range(EPOCHS):
    model_2.backpropagation(X_train, y_train)
    predictions = model_3(torch.from_numpy(X_train))
    _loss: nn.MSELoss = criterion(predictions, torch.from_numpy(y_train))
    optimizer_2.zero_grad()
    _loss.backward()
    optimizer_2.step()
    losses_3.append(_loss.detach().numpy())
    losses_2.append(model_2.get_loss(X[TRAIN_SPLIT:].reshape((X[TRAIN_SPLIT:].shape[0], -1)), y[TRAIN_SPLIT:].reshape((y[TRAIN_SPLIT:].shape[0], -1))))
    optimizer.backpropagation(X_train, y_train)
    #print(model.get_state_dict())
    #optimizer_deep.backpropagation(X_train[i], y_train[i])
    losses.append(loss(Y=y[TRAIN_SPLIT:].reshape((y[TRAIN_SPLIT:].shape[0], -1)), pred=model.forward(X[TRAIN_SPLIT:].reshape((X[TRAIN_SPLIT:].shape[0], -1)))))
    losses_4.append(loss(Y=y[TRAIN_SPLIT:].reshape((y[TRAIN_SPLIT:].shape[0], -1)), pred=model_4.forward(X[TRAIN_SPLIT:].reshape((X[TRAIN_SPLIT:].shape[0], -1)))))

print(z)
plt.plot(np.arange(len(losses)), losses)
plt.plot(np.arange(len(losses_2)), losses_2)
plt.plot(np.arange(len(losses_3)), losses_3)
plt.plot(np.arange(len(losses_4)), losses_4)
plt.legend(["shallow net", "model 2", "toech", "deep net"])
plt.grid(True)
plt.show()

plt.plot(X, y)
plt.plot(X, model.forward(X.reshape((X.shape[0], -1))))
plt.plot(X, model_2.forward(X.reshape((X.shape[0], -1))))
plt.plot(X, model_3.forward(torch.from_numpy(X.reshape((X.shape[0], -1)))).detach().numpy())
plt.plot(X, model_4.forward(X.reshape((X.shape[0], -1))))
plt.legend(["truth", "shalow net", "model2", "torch", "deep net"])
plt.grid(True)
plt.show()
