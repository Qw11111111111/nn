from neural_networks.loss import MSELoss
import numpy as np

dat = np.array([[1, 1, 2], [2, 3, 4], [1, 1, 2], [2, 3, 4]])
dat2 = dat + 1
loss = MSELoss()
print(dat2 - dat)
print(loss.get_grad(dat2, dat))

print(np.dot(np.logical_not(dat2 < 0), dat.T))
z = np.diag(np.linalg.svd(dat.T, full_matrices=True)[1])
l = np.diag(np.diag(dat))
print(z.shape, dat.shape, l.shape)
print(z)
print(l)
print(np.dot(dat, z))
print(np.dot(dat, l))
from neural_networks.loss import MSELoss
import numpy as np

loss = MSELoss()
array_ = np.array([[0, 0.2, 3, 4], [-0.1,-2,3,-0]])
array_2 = np.ones(array_.shape)

def loss_2(pred, Y, axis = 0):
    print(Y.shape[axis])
    return np.sum(np.square(np.sum([pred, - Y], axis = 0))) / Y.shape[axis]

def get_grad_2(y, pred, axis = 0):
    return np.sum(np.dot(2, np.sum([pred, - y], axis = 0)), axis = axis) / y.shape[int(not axis)]
    return np.sum([2 * (pred[i] - y[i]) for i in range(len(y))], axis=1) / len(y)

def normalize():
    array_ = np.array([[0, 2, 3, 4], [-1,-2,3,-0]])
    array_2 = np.ones(array_.shape)
    array_3 = np.array([0.1 for _ in range(array_.shape[0] * array_.shape[1])]).reshape(array_.shape)
    ar_2 = np.exp(array_)
    ar_2 = np.amin([ar_2, np.ones(array_.shape)], axis = 0) 
    ar_3 = - np.amin([(np.ones_like(ar_2) - ar_2) * ( ar_2 +  np.ones_like(ar_2)), array_3], axis = 0)
    ar_4 = ar_3 + np.ones_like(ar_2) + 10 * np.ones_like(ar_2) * ar_3
    print(ar_4)

def normalize(array, alpha):
    #ones = np.ones_like(array)
    array_3 = np.array([alpha for _ in range(array_.shape[0] * array_.shape[1])]).reshape(array_.shape)
    #ar_2 = np.amin([np.exp(array), np.ones(array_.shape)], axis = 0)
    #print((ar_3 := - np.amin([(ones - (ar_2 := np.amin([np.exp(array), ones], axis = 0))) * (ar_2 + ones), array_3], axis = 0)) + ones + 1 / alpha * ones * ar_3)
    return (ar_3 := - np.amin([((ones := np.ones_like(array)) - (ar_2 := np.amin([np.exp(array), ones], axis = 0))) * (ar_2 + ones), (np.zeros_like(array) + alpha)], axis = 0)) + ones + 1 / alpha * ones * ar_3, 0, 0 

array_3 = np.array([0.1 for _ in range(array_.shape[0] * array_.shape[1])]).reshape(array_.shape)

ar_1 =np.amax([array_, np.zeros(array_.shape)], axis = 0) 
ar_2 = np.exp(array_)
ar_2 = np.amin([ar_2, np.ones(array_.shape)], axis = 0) 
ar_3 = - np.amin([(np.ones_like(ar_2) - ar_2) * ( ar_2 +  np.ones_like(ar_2)), array_3], axis = 0)
#* 2  - ar_2 + np.ones_like(array_) - ar_2
ar_4 = ar_3 + np.ones_like(ar_2) + 10 * np.ones_like(ar_2) * ar_3
print(ar_1)
print(ar_2)
print(ar_3)
print(ar_4)
print(np.amax([array_, np.zeros(array_.shape)], axis = 0) - np.amax([array_, array_3], axis = 0))
print(normalize(array_, 0.1)[0])

"""print(np.reshape(array_, -1))
print(get_grad_2(array_, array_2))
print(loss.get_grad(array_, array_2))
print(loss_2(array_2, array_, 1))
print(loss(array_.T, array_2.T))"""
year = 1900
print(year % 4 == 0 and (not year % 100 == 0 or year % 400 == 0))
ar_1 = np.array(([1,2], [3, 4]))
ar_2 = np.array(([[1, 2]]))
print(ar_1.shape,  ar_2.shape)
print(np.dot(ar_2, ar_1))
assert False
zeros = np.zeros(array_.shape)
print(np.amax([array_, zeros], axis = 0))
print(array_.all(where = array_ >= 0, axis = 0, keepdims=True))
#print(all(np.amax([array_, zeros], axis = 0)))
print(np.dot(array_, 4))
#print(np.amin([np.amax([array_, zeros], axis = 0), np.]))