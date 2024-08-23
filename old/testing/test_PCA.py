import numpy as np
import sklearn.decomposition as dec
import matplotlib.pyplot as plt
from utils.maths import PCA
from sklearn import datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_components", action="store", type=int, help="number of resulting components from PCA", default=1)
args = parser.parse_args()

dataset = datasets.load_iris()
labels = dataset.target
X = dataset.data

print(f"{X.shape = }")
print(X[:5])
print(f"{labels.shape = }")
print(labels[:5])

n_components = args.n_components
if n_components < 1: 
    n_components = None

pca = dec.PCA(n_components=n_components)

sklearn_res = pca.fit_transform(X)
PCA_res = PCA(X, n_components=n_components)

print(sklearn_res[:5])
print(f"{sklearn_res.shape = }")
print(sklearn_res[:5])
print(f"{PCA_res.shape = }")
print(PCA_res[:5])

if n_components == 1:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(sklearn_res[:, 0], 
                np.zeros_like(sklearn_res[:, 0]),
                c=labels)
    ax[0].set_title("sklearn plot")
    plt.xlabel("first component of PCA")
    ax[1].scatter(PCA_res[:, 0], 
                np.zeros_like(PCA_res[:, 0]),
                c=labels)
    ax[1].set_title("PCA using utils.maths.PCA plot")
    plt.xlabel("first component of PCA")
    fig.suptitle(f"Plots for reducing to {n_components}-D with PCA")
    plt.show()
elif n_components == 2:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(sklearn_res[:, 0], 
                sklearn_res[:, 1],
                c=labels)
    ax[0].set_title("sklearn plot")
    plt.xlabel("first component of PCA")
    plt.ylabel("second component of PCA")
    ax[1].scatter(PCA_res[:, 0], 
                PCA_res[:, 1],
                c=labels)
    ax[1].set_title("PCA using utils.maths.PCA plot")
    plt.xlabel("first component of PCA")
    plt.ylabel("second component of PCA")
    fig.suptitle(f"Plots for reducing to {n_components}-D with PCA")
    plt.show()