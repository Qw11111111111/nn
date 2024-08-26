import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import sklearn.cluster as clus
from sklearn.metrics import silhouette_score
import argparse
from src.unsupervised import AgglomerativeClusterer, DBScan, KMeans
from src.maths import center_scale
from src.utils import argwhere


parser =  argparse.ArgumentParser()
parser.add_argument("-n", "--n_centers", action="store", type=int, default=2, help="number of centers given to kmeans and agglomerative clusterers. If set to 0, optimal number of centers will be calculated")
parser.add_argument("-N", "--true_centers", action="store", type=int, default=2, help="true number of centers. should be > 0")
parser.add_argument("-r", "--restarts", action="store", type=int, default=1)
parser.add_argument("-v", "--verbose", action="store_true", default=False)
parser.add_argument("-e", "--epsilon", action="store", type=float, default=6e-1)
parser.add_argument("-a", "--no_agglo", action="store_false", default=True)
args = parser.parse_args()

centers = args.n_centers if args.n_centers > 0 else None
true_centers = args.true_centers
restarts = args.restarts
verbose = args.verbose
epsilon = args.epsilon
agglo = not args.no_agglo


X, Y, true_centroids = make_blobs(true_centers * 20, cluster_std=1., random_state=42, return_centers=True, centers=true_centers)


X, mean, std = center_scale(X, verbose=True)
true_centroids = (true_centroids - mean) / std
colors = []

for i in range(20 + (centers if centers is not None else true_centers)):
    colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))


print("kmeans++")
kmeans = KMeans(centers, n_retries=restarts, verbose=verbose, init_method="kmeans++", max_clusters=10, good_score=0.8, scale=False)
assignments, centroids = kmeans.fit_predict(X)
init_c_pp = kmeans.best_initial_centroids
nums = [argwhere(assignments, i, axis=1)[0] for i in range(X.shape[0])]

print("random_choice")
kmeans_random_choice = KMeans(centers, n_retries=restarts, verbose=verbose, init_method="random_choice")
assignments_rc, centroids_rc = kmeans_random_choice.fit_predict(X)
init_c_r_c = kmeans_random_choice.best_initial_centroids
nums_rc = [argwhere(assignments_rc, i, axis=1)[0] for i in range(X.shape[0])]

print("random")
kmeans_random = KMeans(centers, n_retries=restarts, verbose=verbose, init_method="random")
assignments_r, centroids_r = kmeans_random.fit_predict(X)
init_c_r = kmeans_random.best_initial_centroids
nums_r = [argwhere(assignments_r, i, axis=1)[0] for i in range(X.shape[0])]

if agglo:
    print("agglo")
    agglo = AgglomerativeClusterer(clusters=centers if centers is not None else true_centers)
    prox = agglo.fit_predict(X)
    nums_a = [argwhere(prox, i, axis=1)[0] for i in range(X.shape[0])]

print("dbscan")
dbscan = DBScan(epsilon=epsilon, min_pts=20)
color_assignments_db = dbscan.fit_predict(X)



color_assignments = [nums[i] for i in range(X.shape[0])]
color_assignments_r = [nums_r[i] for i in range(X.shape[0])] 
color_assignments_rc = [nums_rc[i] for i in range(X.shape[0])] 
if agglo:
    color_assignments_ag = [nums_a[i] for i in range(X.shape[0])]

s_score = kmeans.appr_silhouette(X)
s_score_2 = silhouette_score(X, np.array(color_assignments))
print(f"silhouette scores of kmeans++: my implementation | sklearn \n{f"{s_score:.3f} | {s_score_2:.3f}":>56}")

kmean_sk = clus.KMeans(centers if centers is not None else true_centers, init="k-means++")
labels_sk = kmean_sk.fit_predict(X)

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
            c=[colors[i] for i in Y])
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