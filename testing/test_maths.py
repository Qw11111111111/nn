import numpy as np 
import sklearn.decomposition as dec
import matplotlib.pyplot as plt
from utils.maths import PCA
from sklearn import datasets


def pca_comparison(X, n_components, labels):
  """X: Standardized dataset, observations on rows
     n_components: dimensionality of the reduced space
     labels: targets, for visualization
  """

  """ # numpy
  # -----

  # calculate eigen values
  X_cov = np.cov(X.T)
  e_values, e_vectors = np.linalg.eigh(X_cov)

  # Sort eigenvalues and their eigenvectors in descending order
  e_ind_order = np.flip(e_values.argsort())
  e_values = e_values[e_ind_order]
  e_vectors = e_vectors[:, e_ind_order] # note that we have to re-order the columns, not rows

  # now we can project the dataset on to the eigen vectors (principal axes)
  prin_comp_evd = X @ e_vectors"""


X = np.linspace(0, 20, 100) + np.random.normal(size = 100)
Y = 2 * X + np.random.normal(size = 100)

dataset = datasets.load_iris()
labels = dataset.target
X = dataset.data
"""plt.scatter(X, Y)
plt.show()
"""
pca = dec.PCA(n_components=1)

prin_comp_sklearn = pca.fit_transform(X)

prin_comp_evd = PCA(X, n_components=1)

print(prin_comp_sklearn[:10])
print(prin_comp_sklearn.shape)
print(prin_comp_evd[:10])
print(prin_comp_evd.shape)
n_components = 1


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(prin_comp_sklearn[:, 0], 
            np.zeros_like(prin_comp_sklearn[:, 0]),
            c=labels)
ax[0].set_title("sklearn plot")
ax[1].scatter(prin_comp_evd[:, 0], 
            np.zeros_like(prin_comp_evd[:, 0]),
            c=labels)
ax[1].set_title("PCA using EVD plot")
fig.suptitle(f"Plots for reducing to {n_components}-D")
plt.show()