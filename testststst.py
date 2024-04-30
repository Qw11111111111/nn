from loss_functions.main import MSELoss
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