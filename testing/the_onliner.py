import torch
from torch import nn
import matplotlib.pyplot as plt

"""torch.manual_seed(42)
# normal code
model_2 = nn.Sequential(nn.Linear(1,1,False), nn.ReLU(), nn.Linear(1,1,False))
crit_2 = nn.MSELoss()
optimizer_2 = torch.optim.SGD(model_2.parameters(), 3e-7)
X = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0]).unsqueeze(0).T
y = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).unsqueeze(0).T
for ep in range(200):
    pred = model_2(X)
    loss = crit_2(pred, y)
    optimizer_2.zero_grad()
    loss.backward()
    optimizer_2.step()

print(model_2.forward(X))
"""
[[[(x := lambda: print("hello"))() if True else None],[(x := lambda: print("du"))() if True else None],[(x := lambda: print("menshc"))() if True else None]] for i in range(1)]

# the above code compressed to one line
[[[torch.manual_seed(42)], [(train := lambda model, crit, optimizer, X, y: [[[(zero := lambda: optimizer.zero_grad())() if True else None], [(get_loss := lambda: crit(model(X), y).backward())() if True else None], [(step := lambda: optimizer.step())() if True else None]] for ep in range(200)])((model := nn.Sequential(nn.Linear(1,1,False), nn.ReLU(), nn.Linear(1,1,False))), nn.MSELoss(), torch.optim.SGD(model.parameters(), 6e-7), X := torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0]).unsqueeze(0).T, y := torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).unsqueeze(0).T) if True else None], [(printer := lambda: print(model.forward(X)))() if True else None], [(plotter := lambda: plt.plot(model.forward(X).detach().numpy(), y.detach().numpy()))() if True else None], [(show := lambda: plt.show())() if True else None]] for _ in range(1)]

assert False
(train := lambda model, crit, optimizer, X, y: [(zero := lambda optimizer: optimizer.zero_grad())(optimizer)(get_loss := lambda model, X: crit(model(X), y).backward())(model, X, y)(step := lambda optimizer: optimizer.step())(optimizer) for ep in range(10)])(model := nn.Sequential(nn.Linear(1,1,False), nn.ReLU(), nn.Linear(1,1,False)), nn.MSELoss(), torch.optim.SGD(model.parameters(), 1e-5), torch.tensor([100, 200, 300]), torch.tensor([1, 2, 3])) if True else None
print(model(100))

(x := lambda y: print("hello"))(0) if  True else None