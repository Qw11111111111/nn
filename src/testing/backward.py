import jax.numpy as jnp
from  jax import grad, jacobian
from src.loss import MSELoss
from src.layers import LinearLayer
from src.models import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

array_ = jnp.arange(10.)

def dot(arr1):
  return jnp.square(arr1)

array_ = dot(array_)
print(array_)

deriv = jacobian(dot)
print(deriv)

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(5.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))

print(deriv(jnp.array([10.0, 5.0, 3.0, 42.0]).reshape((2, 2))))

loss_ = MSELoss()

X = jnp.arange(5.)

Y = jnp.arange(5.) + 1.0

error = loss_(X, Y)

print(error)

gradient = loss_.backward(X, Y)

print(gradient)

print("done")

X = jnp.arange(25.).reshape((5, 5))

Y = jnp.arange(5.).reshape((-1, 1))

linlay = LinearLayer((5, 1), bias=False)
out = linlay(X)

print(out)

err = loss_(out, Y)

print(err)

gradi = loss_.backward(out, Y)


print(gradi)

grad_2 = linlay.backward(1.)

print(grad_2)


model = LinearLayer((1, 1), False)
#model.weights = -model.weights if model.weights[0] > 0 else model.weights
criterion = MSELoss()

model2 = LinearRegression((1, 1), False)

X = jnp.arange(50.).reshape((-1, 1))

Y = jnp.arange(50.) * 2

Y += np.random.normal(0., 2., Y.shape) + 0
Y = Y.reshape((-1, 1))

print(X.shape, Y.shape)

pred = model(X)
pred2 = model2(X)

X_p = X[:,0]

plt.plot(X_p, Y)
plt.plot(X_p, pred)
plt.plot(X_p, pred2)
plt.legend(["truth", "pred", "pred2"])
plt.show()

errors = []
last = 0
batch = 10
print("weights: ", model.weights)
print("bias: ", model.bias)
for i in range(200):
  random_idx = np.random.randint(0, X.shape[0] - batch)
  r_X, r_Y = X[random_idx:random_idx + batch], Y[random_idx:random_idx + batch]
  #print(r_X.shape, r_X)
  pred = model(r_X)
  errors.append(criterion(pred, r_Y))
  d_error = criterion.backward(pred, r_Y).reshape((1, -1))
  #print(d_error.shape)
  d_pred = model.backward(r_X)#.reshape((1, batch, batch))
  #print(len(d_pred))
  #print("pred",d_pred.shape)
  #print(d_pred)
  #print("err", d_error.shape)
  ###d_w, d_b = jnp.average(d_pred[0], axis = 0), jnp.average(d_pred[1], axis = 0)
  #print(d_error.shape)
  #print("w, b",d_w.shape, d_b.shape)
  #print("w",d_w,"b", d_b)
  #assert False
  #print(d_w[0][0].reshape((-1, 1)).shape)
  #print("weights", model.weights.shape)
  ###model.weights -= 1e-3 * jnp.dot(d_error, d_w).T
  #print("weights", model.weights.shape)
  #assert False
  
  model.weights -= 5e-3 * jnp.average(jnp.dot(d_error, d_pred)).reshape(model.weights.shape) + 1e-1 * last
  #model.bias -= 1e-3 * jnp.dot(d_error, jnp.average(d_b)) + 0 * last
  #last = 5e-2 * jnp.average(d_error)
  if i > 4  and errors[-1] > errors[-2] and errors[-1] > errors[-3]:
    continue


#model2.fit(optim = None, criterion=criterion, X=X, Y=Y)

plt.plot(jnp.arange(i + 1), errors)
plt.show()
print("weights: ", model.weights)
print("bias: ", model.bias)

pred = model(X)
pred2 = model2(X)
plt.plot(X_p, Y)
plt.plot(X_p, pred)
plt.plot(X_p, pred2)
plt.legend(["truth", "pred", "pred2"])
plt.show()
