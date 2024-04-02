from loss_functions.main import MSELoss
import numpy as np

loss = MSELoss()
array_ = np.array([[1, 2, 3, 4], [1,2,3,4]])
array_2 = np.ones(array_.shape)

def loss_2(pred, Y, axis = 0):
    print(Y.shape[axis])
    return np.sum(np.square(np.sum([pred, - Y], axis = 0))) / Y.shape[axis]

def get_grad_2(y, pred, axis = 0):
    return np.sum(np.dot(2, np.sum([pred, - y], axis = 0)), axis = axis) / y.shape[int(not axis)]
    return np.sum([2 * (pred[i] - y[i]) for i in range(len(y))], axis=1) / len(y)

print(get_grad_2(array_, array_2))
print(loss.get_grad(array_, array_2))
print(loss_2(array_2, array_, 1))
print(loss(array_.T, array_2.T))
assert False
zeros = np.zeros(array_.shape)
print(np.amax([array_, zeros], axis = 0))
print(array_.all(where = array_ >= 0, axis = 0, keepdims=True))
#print(all(np.amax([array_, zeros], axis = 0)))
print(np.dot(array_, 4))
#print(np.amin([np.amax([array_, zeros], axis = 0), np.]))