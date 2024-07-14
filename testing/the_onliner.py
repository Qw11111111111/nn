import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
def long_line():
    [  torch.manual_seed(42),  val_losses := [],  train_losses := [],  train := lambda model, crit, optimizer, X, y, verbose, val_split: [ [  optimizer.zero_grad(),loss := crit(model(X[:val_split]), y[:val_split]),  loss.backward(),optimizer.step(),  train_losses.append(loss.detach().numpy()) if verbose else None,  val_losses.append(crit(model(X[val_split:]), y[val_split:]).detach().numpy()) if verbose else None, ] for _ in range(50)],  validate := lambda model, criterion, X, y: criterion(model.forward(X), y),  best_lr := 0,  minimum := torch.inf,  lrs := [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  X := torch.linspace(-100, 100, 500),y := X * 2 + torch.normal(0, 100, X.shape),  rand_perm := torch.randperm(X.shape[0]),  original_X := torch.empty_like(X).copy_(X).unsqueeze(0).T,  X := torch.tensor([X[i] for i in rand_perm]).unsqueeze(0).T,  y := torch.tensor([y[i] for i in rand_perm]).reshape(X.shape),  val_split := int(0.8 * X.shape[0]),  model_factory := lambda : nn.Sequential(nn.Linear(1, 100, False),nn.ReLU(),nn.Linear(100, 100, False),nn.ReLU(),nn.Linear(100, 100, False),nn.ReLU(),nn.Linear(100, 1, False)  ),  optim_factory := lambda model, lr: torch.optim.Adam(model.parameters(), lr),  criterion := nn.MSELoss(),  weight_reset := lambda m: m.reset_parameters() if callable(getattr(m, "reset_parameters", None)) else None,  [[ train(model := model_factory().train(),criterion, optim_factory(model, lr), X, y,False,val_split  ), [  best_lr := lr,  minimum := loss ] if (loss := validate(model, criterion, X[val_split:], y[val_split:])) < minimum else None] for lr in lrs  ],  model.apply(weight_reset),  print(f"{best_lr = }"),  train(model, criterion, optim_factory(model, best_lr), X, y, True, val_split),  plt.plot(np.arange(len(train_losses)), train_losses),  plt.plot(np.arange(len(val_losses)), val_losses),  plt.xlabel("epoch"),  plt.ylabel("loss"),  plt.title("losses"),  plt.legend(["train loss", "test loss"]),  plt.show(),  plt.plot(X_np := original_X.detach().numpy(), model.forward(original_X).detach().numpy()),plt.scatter(X.detach().numpy(), y.detach().numpy(), 1),plt.plot(X_np, X_np * 2),plt.ylabel("Y"),plt.xlabel("X"),plt.title("Prediction of MLP"),plt.legend(["MLP", "data", "truth"]),plt.show()]

from utils.utils import timeit

# normal code
@timeit
def normal():
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
@timeit
def one_line():
    [torch.manual_seed(42), (_ := lambda model, crit, optimizer, X, y: [[optimizer.zero_grad(), crit(model(X), y).backward(), optimizer.step()] for _ in range(60)])((model := nn.Sequential(nn.Linear(1,10,False), nn.ReLU(), nn.Linear(10,1,False))), nn.MSELoss(), torch.optim.SGD(model.parameters(), 1e-8), X := torch.linspace(0, 1000, 500).unsqueeze(0).T, y := X * 10 + torch.normal(0, 100, X.shape)), print(model.forward(X[:10])), print(X.shape), plt.plot(X_np := X.detach().numpy(), model.forward(X).detach().numpy()), plt.plot(X_np, y.detach().numpy()), plt.plot(X_np, X_np * 10), plt.ylabel("Y"), plt.xlabel("X"), plt.title("Prediction of Shallow Net"), plt.legend(["shallow net", "data", "truth"]), plt.show()]

@timeit
def one_line_expanded():
    [
    torch.manual_seed(42), 
    (_ := lambda model, crit, optimizer,
        X, y: [
                [
                    optimizer.zero_grad(), 
                    crit(model(X), y).backward(), 
                    optimizer.step()
                ] for _ in range(60)
            ])(
                model := nn.Sequential(
                                        nn.Linear(1,10,False), 
                                        nn.ReLU(), 
                                        nn.Linear(10,1,False)
                                        ), 
                        nn.MSELoss(), 
                        torch.optim.SGD(model.parameters(), 1e-8), 
                        X := torch.linspace(0, 1000, 500).unsqueeze(0).T, 
                        y := X * 10 + torch.normal(0, 100, X.shape)
                ), 
    print(model.forward(X[:10])), 
    plt.plot(X_np := X.detach().numpy(), model.forward(X).detach().numpy()), 
    plt.plot(X_np, y.detach().numpy()), 
    plt.plot(X_np, X_np * 10), 
    plt.ylabel("Y"), 
    plt.xlabel("X"), 
    plt.title("Prediction of Shallow Net"), 
    plt.legend(["shallow net", "data", "truth"]), 
    plt.show()
    ]

"""
[
    torch.manual_seed(42),
    train := lambda x: [
        x
    ],
    params := 0,
    validate := lambda z : [
        z
    ],
    [
        [
            train(x := 1) for _ in range(10)
        ],
        [
            [
                params := 2
            ] if True else None
        ]
        for lr in [1,2,3]
    ]
]"""

@timeit
def one_line_expanded_ultimate():
    return [
        torch.manual_seed(42),
        val_losses := [],
        train_losses := [],
        train := lambda model, crit, optimizer, X, y, verbose, val_split: 
            [   
                [
                    optimizer.zero_grad(), 
                    loss := crit(model(X[:val_split]), y[:val_split]),
                    loss.backward(), 
                    optimizer.step(),
                    train_losses.append(loss.detach().numpy()) if verbose else None,
                    val_losses.append(crit(model(X[val_split:]), y[val_split:]).detach().numpy()) if verbose else None,
                ] for _ in range(50)
            ],
        validate := lambda model, criterion, X, y: criterion(model.forward(X), y),
        best_lr := 0,
        minimum := torch.inf,
        lrs := [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        X := torch.linspace(-100, 100, 500), 
        y := X * 2 + torch.normal(0, 100, X.shape),
        rand_perm := torch.randperm(X.shape[0]),
        original_X := torch.empty_like(X).copy_(X).unsqueeze(0).T,
        X := torch.tensor([X[i] for i in rand_perm]).unsqueeze(0).T,
        y := torch.tensor([y[i] for i in rand_perm]).reshape(X.shape),
        val_split := int(0.8 * X.shape[0]),
        model_factory := lambda : nn.Sequential(
            nn.Linear(1, 100, False),
            nn.ReLU(),
            nn.Linear(100, 100, False),
            nn.ReLU(),
            nn.Linear(100, 100, False),
            nn.ReLU(),
            nn.Linear(100, 1, False)
        ),
        optim_factory := lambda model, lr: torch.optim.Adam(model.parameters(), lr),
        criterion := nn.MSELoss(),
        weight_reset := lambda m: m.reset_parameters() if callable(getattr(m, "reset_parameters", None)) else None,
        [
            [
                train(
                        model := model_factory().train(),
                        criterion, 
                        optim_factory(model, lr), 
                        X, 
                        y,
                        False,
                        val_split
                    ),
                [
                    best_lr := lr,
                    minimum := loss
                ] if (loss := validate(model, criterion, X[val_split:], y[val_split:])) < minimum else None
            ] for lr in lrs
        ],
        model.apply(weight_reset),
        print(f"{best_lr = }"),
        train(model, criterion, optim_factory(model, best_lr), X, y, True, val_split),
        plt.plot(np.arange(len(train_losses)), train_losses),
        plt.plot(np.arange(len(val_losses)), val_losses),
        plt.xlabel("epoch"),
        plt.ylabel("loss"),
        plt.title("losses"),
        plt.legend(["train loss", "test loss"]),
        plt.show(),
        plt.plot(X_np := original_X.detach().numpy(), model.forward(original_X).detach().numpy()), 
        plt.scatter(X.detach().numpy(), y.detach().numpy(), 1), 
        plt.plot(X_np, X_np * 2), 
        plt.ylabel("Y"), 
        plt.xlabel("X"), 
        plt.title("Prediction of MLP"), 
        plt.legend(["MLP", "data", "truth"]), 
        plt.show()]

#one_line()
#normal()
#one_line()
#_ = one_line_expanded_ultimate()
long_line()