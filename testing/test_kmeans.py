from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import sklearn.cluster as clus
from utils.maths import center_scale
from algorithms.clustering import KMeans, AgglomerativeClusterer, DBScan
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import argwhere, timeit
from random import randint


parser =  argparse.ArgumentParser()
parser.add_argument("-n", "--n_centers", action="store", type=int, default=2)
parser.add_argument("-N", "--true_centers", action="store", type=int, default=2)
parser.add_argument("-r", "--restarts", action="store", type=int, default=1)
parser.add_argument("-v", "--verbose", action="store_true", default=False)
parser.add_argument("-e", "--epsilon", action="store", type=float, default=4e-1)
parser.add_argument("-a", "--no_agglo", action="store_false", default=True)
args = parser.parse_args()

centers = args.n_centers if args.n_centers > 0 else None
true_centers = args.true_centers
restarts = args.restarts
verbose = args.verbose
epsilon = args.epsilon
agglo = args.no_agglo

X, y, true_centroids = make_blobs(centers=centers if centers is not None else true_centers, n_samples=centers * 20 if centers is not None else 20 * true_centers, cluster_std=1, return_centers=True)

X, mean, std = center_scale(X, verbose=True)
true_centroids = (true_centroids - mean) / std
colors = []

for i in range(20 + (centers if centers is not None else true_centers)):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

print("kmeans++")
kmeans = KMeans(centers, n_retries=restarts, verbose=verbose, init_method="kmeans++", max_clusters=10, good_score=0.8, scale=False)
assignments, centroids = kmeans.fit_predict(X)
init_c_pp = kmeans.best_initial_centroids
#print(centroids)
#print(init_c_pp)
print(len(init_c_pp) == len(centroids))
print("random_choice")
kmeans_random_choice = KMeans(centers, n_retries=restarts, verbose=verbose, init_method="random_choice")
assignments_rc, centroids_rc = kmeans_random_choice.fit_predict(X)
init_c_r_c = kmeans_random_choice.best_initial_centroids
print(len(init_c_r_c) == len(centroids_rc))
#print(centroids_rc)
#print(init_c_r_c)
print("random")
kmeans_random = KMeans(centers, n_retries=restarts, verbose=verbose, init_method="random")
assignments_r, centroids_r = kmeans_random.fit_predict(X)
init_c_r = kmeans_random.best_initial_centroids
print(len(init_c_r) == len(centroids_r))
#print(centroids_r)
#print(init_c_r)
if agglo:
    print("agglo")
    agglo = AgglomerativeClusterer(clusters=10)#centers if centers is not None else true_centers)

    prox = agglo.fit_predict(X)
    #print(prox, "prox")
    #print(list(reversed(agglo.history)))
print("dbscan")
dbscan = DBScan(epsilon=epsilon, min_pts=10)
color_assignments_db = dbscan.fit_predict(X)
#print(len(color_assignments_db))
#print(color_assignments_db)

nums = [argwhere(assignments, i, axis=1)[0] for i in range(X.shape[0])]
nums_r = [argwhere(assignments_r, i, axis=1)[0] for i in range(X.shape[0])]
nums_rc = [argwhere(assignments_rc, i, axis=1)[0] for i in range(X.shape[0])]
if agglo:
    nums_a = [argwhere(prox, i, axis=1)[0] for i in range(X.shape[0])]

color_assignments = [nums[i] for i in range(X.shape[0])]
color_assignments_r = [nums_r[i] for i in range(X.shape[0])] 
color_assignments_rc = [nums_rc[i] for i in range(X.shape[0])] 
if agglo:
    color_assignments_ag = [nums_a[i] for i in range(X.shape[0])]

s_score = kmeans.appr_silhouette(X)
s_score_2 = silhouette_score(X, np.array(color_assignments))
print(f"silhouette scores of kmeans++: my implementation | sklearn \n{f"{s_score:.3f} | {s_score_2:.3f}":>56}")

kmean_sk = clus.KMeans(centers if centers is not None else 2, init="k-means++")
labels_sk = kmean_sk.fit_predict(X)
print(len(labels_sk))

agglo_sk = clus.AgglomerativeClustering(centers if centers is not None else true_centers)
labels_agglom = agglo_sk.fit_predict(X)

dbscan_sk = clus.DBSCAN()
labels_dbscn = dbscan_sk.fit_predict(X)

fig, ax = plt.subplots(2, 5, figsize=(25, 10))
ax[0][0].scatter(centroids[:][:,0], centroids[:][:,1], marker = "P", c = "red")
ax[0][0].scatter(init_c_pp[:][:,0], init_c_pp[:][:,1], marker = "D", c = "red")
ax[0][0].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in color_assignments])
ax[0][0].set_title("my kmeans kmeans++")
ax[0][1].scatter(centroids_r[:][:,0], centroids_r[:][:,1], marker = "P", c = "red")
ax[0][1].scatter(init_c_r[:][:,0], init_c_r[:][:,1], marker = "D", c = "red")
ax[0][1].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in color_assignments_r])
ax[0][1].set_title("my kmeans random")
ax[0][2].scatter(centroids_rc[:][:,0], centroids_rc[:][:,1], marker = "P", c = "red")
ax[0][2].scatter(init_c_r_c[:][:,0], init_c_r_c[:][:,1], marker = "D", c = "red")
ax[0][2].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in color_assignments_rc])
ax[0][2].set_title("my kmeans random choice")
ax[0][3].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in labels_agglom])
ax[0][3].set_title("sklearn kmeans kmeans++")
ax[0][4].scatter(true_centroids[:][:,0], true_centroids[:][:,1], marker = "P", c = "red")
ax[0][4].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in y])
ax[0][4].set_title("true blobs")


if agglo:
    ax[1][0].scatter(X.T[:][0], X.T[:][1],
                c=[colors[i] for i in color_assignments_ag])
    ax[1][0].set_title("my agglo")
ax[1][1].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in color_assignments_db])
ax[1][1].set_title("my dbscan")

legend = np.array([i for i in range(len(colors))])
ax[1][2].scatter(legend, np.zeros_like(legend),
                 c = [colors[i] for i in range(len(colors))])
ax[1][2].set_title("colors")

ax[1][3].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in labels_agglom])
ax[1][3].set_title("sklearn agglo")
ax[1][4].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in labels_dbscn])
ax[1][4].set_title("sklearn dbscan")



fig.suptitle(f"Plots for Kmeans clustering with my and sklearn implementation")
plt.show()

"""fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in labels_agglom])
ax[0].set_title("sklearn agglo plot")
ax[1].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in color_assignments_ag])
ax[1].set_title("my agglo plot")
ax[2].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in color_assignments_db])
ax[2].set_title("my dbscan plot")
ax[3].scatter(X.T[:][0], X.T[:][1],
            c=[colors[i] for i in y])
ax[3].set_title("true blobs")
fig.suptitle(f"Plots for Kmeans clustering with my and sklearn implementation")
plt.show()
"""