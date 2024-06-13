import torch
from torch import nn
import matplotlib.pyplot as plt

# normal code
torch.manual_seed(42)
model_2 = nn.Sequential(nn.Linear(1,1,False), nn.ReLU(), nn.Linear(1,1,False))
crit_2 = nn.MSELoss()
optimizer_2 = torch.optim.SGD(model_2.parameters(), 6e-7)
X = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0]).unsqueeze(0).T
y = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).unsqueeze(0).T
for _ in range(200):
    pred = model_2(X)
    loss = crit_2(pred, y)
    optimizer_2.zero_grad()
    loss.backward()
    optimizer_2.step()

print(model_2.forward(X))

plt.plot(X.detach().numpy(), model_2.forward(X).detach().numpy())
plt.plot(X.detach().numpy(), y.detach().numpy())
plt.legend(["Shallow net", "truth"])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Prediction of Shallow Net")
plt.show()

# the above code compressed to one line
[torch.manual_seed(42), (train := lambda model, crit, optimizer, X, y: [[optimizer.zero_grad(), crit(model(X), y).backward(), optimizer.step()] for _ in range(200)])((model := nn.Sequential(nn.Linear(1,1,False), nn.ReLU(), nn.Linear(1,1,False))), nn.MSELoss(), torch.optim.SGD(model.parameters(), 6e-7), X := torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0]).unsqueeze(0).T, y := torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).unsqueeze(0).T), print(model.forward(X)), plt.plot(X.detach().numpy(), model.forward(X).detach().numpy()), plt.plot(X.detach().numpy(), y.detach().numpy()), plt.ylabel("Y"), plt.xlabel("X"), plt.title("Prediction of Shallow Net"), plt.legend(["shallow net", "truth"]), plt.show()]