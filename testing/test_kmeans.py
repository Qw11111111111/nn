from sklearn.datasets import make_blobs
from utils.maths import KMeans
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import argwhere
from random import randint


pareser =  argparse.ArgumentParser()
pareser.add_argument("-n", "--n_centers", action="store", type=int, default=2)
args = pareser.parse_args()

centers = args.n_centers

X, y = make_blobs(centers=centers)

colors = []

for i in range(centers):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

plt.scatter(X.T[:][0], X.T[:][1], c=[colors[i] for i in y])
plt.show()


kmeans = KMeans(centers)
assignments, centroids = kmeans.fit_predict(X)
                                                                 
# works only if n = 2:

nums = [argwhere(assignments, i, axis=1)[0] for i in range(X.shape[0])]

print(nums)
color_assignments = [nums[i] for i in range(X.shape[0])]

plt.scatter(X.T[:][0], X.T[:][1], c=[colors[i] for i in color_assignments])
plt.show()
