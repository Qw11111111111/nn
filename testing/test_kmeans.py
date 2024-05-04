from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import sklearn.cluster as clus
from utils.maths import center_scale
from algorithms.clustering import KMeans
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import argwhere, timeit
from random import randint


parser =  argparse.ArgumentParser()
parser.add_argument("-n", "--n_centers", action="store", type=int, default=2)
parser.add_argument("-r", "--restarts", action="store", type=int, default=1)
parser.add_argument("-v", "--verbose", action="store_true", default=False)
args = parser.parse_args()

centers = args.n_centers if args.n_centers > 0 else None
restarts = args.restarts
verbose = args.verbose

X, y = make_blobs(centers=centers, n_samples=centers * 100 if centers is not None else 200)

X = center_scale(X)
colors = []

for i in range(centers if centers is not None else 5):
    colors.append('#%06X' % randint(0, 0xFFFFFF))


kmeans = KMeans(centers, n_retries=restarts, verbose=verbose, init_method="kmeans++", max_clusters=10, good_score=0.9)
assignments, centroids = kmeans.fit_predict(X)

kmeans_random_choice = KMeans(centers, n_retries=restarts, verbose=verbose, init_method="random_choice")
assignments_rc, centroids_rc = kmeans_random_choice.fit_predict(X)

kmeans_random = KMeans(centers, n_retries=restarts, verbose=verbose, init_method="random")
assignments_r, centroids_r = kmeans_random.fit_predict(X)

nums = [argwhere(assignments, i, axis=1)[0] for i in range(X.shape[0])]
nums_r = [argwhere(assignments_r, i, axis=1)[0] for i in range(X.shape[0])]
nums_rc = [argwhere(assignments_r, i, axis=1)[0] for i in range(X.shape[0])]

color_assignments = [nums[i] for i in range(X.shape[0])]
color_assignments_r = [nums_r[i] for i in range(X.shape[0])] 
color_assignments_rc = [nums_rc[i] for i in range(X.shape[0])] 

s_score = kmeans.appr_silhouette(X)

print(s_score, "mine")
s_score_2 = silhouette_score(X, np.array(color_assignments))
print(s_score_2, "slearn")

kmean_sk = clus.KMeans(centers if centers is not None else 3, init="k-means++")
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
ax[3].set_title("sklearn plot")
ax[4].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in y])
ax[4].set_title("true blobs")
fig.suptitle(f"Plots for Kmeans clustering with my and sklearn implementation")
plt.show()

