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

plt.plot(model_2.forward(X).detach().numpy(), y.detach().numpy())
plt.show()

# the concept
[[[(x := lambda: print("hello"))()], [(x := lambda: print("I"))()], [(x := lambda: print("am"))()], [(x := lambda: print("stupid"))()]] for i in range(1)]

# the above code compressed to one line
[[[torch.manual_seed(42)], [(train := lambda model, crit, optimizer, X, y: [[[(zero := lambda: optimizer.zero_grad())()], [(get_loss := lambda: crit(model(X), y).backward())()], [(step := lambda: optimizer.step())()]] for _ in range(200)])((model := nn.Sequential(nn.Linear(1,1,False), nn.ReLU(), nn.Linear(1,1,False))), nn.MSELoss(), torch.optim.SGD(model.parameters(), 6e-7), X := torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0]).unsqueeze(0).T, y := torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).unsqueeze(0).T)], [(printer := lambda: print(model.forward(X)))()], [(plotter := lambda: plt.plot(model.forward(X).detach().numpy(), y.detach().numpy()))()], [(show := lambda: plt.show())()]] for _ in range(1)]