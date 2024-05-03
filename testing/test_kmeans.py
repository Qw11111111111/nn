from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import sklearn.cluster as clus
from utils.maths import KMeans, center_scale
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import argwhere, timeit
from random import randint

#OMP_NUM_THREADS=1

pareser =  argparse.ArgumentParser()
pareser.add_argument("-n", "--n_centers", action="store", type=int, default=2)
pareser.add_argument("-r", "--restarts", action="store", type=int, default=1)
args = pareser.parse_args()

centers = args.n_centers
restarts = args.restarts

X, y = make_blobs(centers=centers, n_samples=centers * 100)

X = center_scale(X)
colors = []

for i in range(centers):
    colors.append('#%06X' % randint(0, 0xFFFFFF))


kmeans = KMeans(centers, n_retries=restarts, verbose=True, init_method="kmeans++")
assignments, centroids = kmeans.fit_predict(X)

kmeans_random_choice = KMeans(centers, n_retries=restarts, verbose=True, init_method="random_choice")
assignments_rc, centroids_rc = kmeans_random_choice.fit_predict(X)

kmeans_random = KMeans(centers, n_retries=restarts, verbose=True, init_method="random")
assignments_r, centroids_r = kmeans_random.fit_predict(X)

nums = [argwhere(assignments, i, axis=1)[0] for i in range(X.shape[0])]
nums_r = [argwhere(assignments_r, i, axis=1)[0] for i in range(X.shape[0])]
nums_rc = [argwhere(assignments_r, i, axis=1)[0] for i in range(X.shape[0])]

color_assignments = [nums[i] for i in range(X.shape[0])]
color_assignments_r = [nums_r[i] for i in range(X.shape[0])] 
color_assignments_rc = [nums_rc[i] for i in range(X.shape[0])] 

s_score = kmeans.silhouette(X)

print(s_score, "mine")
s_score_2 = silhouette_score(X, np.array(color_assignments))
print(s_score_2, "slearn")

kmean_sk = clus.KMeans(centers, init="k-means++")
labels_sk = kmean_sk.fit_predict(X)


fig, ax = plt.subplots(1, 5, figsize=(25, 5))
ax[0].scatter(centroids[:][:,0], centroids[:][:,1], marker = "P")
ax[0].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in color_assignments])
ax[0].set_title("utils.math.kmeans kmeans++ plot")
ax[1].scatter(centroids_r[:][:,0], centroids_r[:][:,1], marker = "P")
ax[1].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in color_assignments_r])
ax[1].set_title("utils.math.kmeans random plot")
ax[2].scatter(centroids_rc[:][:,0], centroids_rc[:][:,1], marker = "P")
ax[2].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in color_assignments_rc])
ax[2].set_title("utils.math.kmeans random choice plot")
ax[3].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in labels_sk])
ax[3].set_title("sklearn plot")#ax[1].scatter(X.T[:][0], X.T[:][1],
ax[4].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in y])
ax[4].set_title("true blobs")
fig.suptitle(f"Plots for Kmeans clustering with my and sklearn implementation")
plt.show()



"""plt.scatter(X.T[:][0], X.T[:][1], c=[colors[i] for i in color_assignments])
plt.show()
"""