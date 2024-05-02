from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import sklearn.cluster as clus
from utils.maths import KMeans
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import argwhere
from random import randint

#OMP_NUM_THREADS=1

pareser =  argparse.ArgumentParser()
pareser.add_argument("-n", "--n_centers", action="store", type=int, default=2)
pareser.add_argument("-r", "--restarts", action="store", type=int, default=1)
args = pareser.parse_args()

centers = args.n_centers
restarts = args.restarts

X, y = make_blobs(centers=centers)

colors = []

for i in range(centers):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

plt.scatter(X.T[:][0], X.T[:][1], c=[colors[i] for i in y])
plt.show()


kmeans = KMeans(centers, n_retries=restarts)
assignments, centroids = kmeans.fit_predict(X)

nums = [argwhere(assignments, i, axis=1)[0] for i in range(X.shape[0])]

color_assignments = [nums[i] for i in range(X.shape[0])]

s_score = kmeans.silhouette(X)
print(s_score)
s_score_2 = silhouette_score(X, np.array(color_assignments))
print(s_score_2)

kmean_sk = clus.KMeans(centers)
labels_sk = kmean_sk.fit_predict(X)



fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in color_assignments])
ax[0].set_title("utils.math.kmeans plot")
ax[1].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in labels_sk])
ax[2].set_title("sklearn plot")#ax[1].scatter(X.T[:][0], X.T[:][1],
ax[2].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in y])
ax[2].set_title("true blobs")
fig.suptitle(f"Plots for Kmeans clustering with my and sklearn implementation")
plt.show()



"""plt.scatter(X.T[:][0], X.T[:][1], c=[colors[i] for i in color_assignments])
plt.show()
"""