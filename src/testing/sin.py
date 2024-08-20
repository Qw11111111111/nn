import numpy as np
import matplotlib.pyplot as plt

def err(X: np.ndarray, Y: np.ndarray) -> float:
    return 1 / X.shape[0] * np.sum(np.square(Y - X))

def fn(X: np.ndarray, theta: float) -> np.ndarray:
    return np.sin(X - theta)

def d_err(X: np.ndarray, Y: np.ndarray, theta: float) -> float:
    return 2 / X.shape[0] * np.sum((Y - fn(X, theta)) * np.cos(X - theta))


X = np.arange(500.).reshape((-1, 1))
print(X.shape)
Y = fn(X, 0.) + np.random.normal(0., .5, X.shape)

estimator = np.random.randint(-5., 5.) + np.random.random()
errs = []

print(estimator)

last = 0
for i in range(10000):
    current = fn(X, estimator)
    errs.append(err(current, Y))
    curr_d_err = d_err(X, Y, estimator)
    update = 5e-5 * curr_d_err + 0.99 * last
    estimator -= update
    last = update

estimator2 = -5.
min = np.inf
best = 0

while estimator2 <= 5.:
    curr_err = err(fn(X, estimator2), Y)
    if curr_err < min:
        min = curr_err
        best = estimator2
    estimator2 += 0.0001

Y_true = fn(X, 0.)

print("GD: ", estimator)
print(err(fn(X, estimator), Y_true))
print("naive: ", best)
print(err(fn(X, best), Y_true))

plt.plot(np.arange(i + 1), errs)
plt.show()

plt.plot(X[:50], Y_true[:50])
plt.plot(X[:50], fn(X[:50], estimator))
plt.plot(X[:50], fn(X[:50], best))
plt.legend(["true", "pred", "pred2"])
plt.show()