import numpy as np
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

X = load_wine().data
time_r = []
shape = []

start_time = time.time()
kmeans1 = KMeans(n_clusters=3)
kmeans1.fit(X)
plt.figure()
plt.title("Clustering for original database")
plt.scatter(X[:, 0], X[:, 1], c=kmeans1.labels_)
plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[
            :, 1], s=200, marker='*', c='red')
end_time = time.time()
time_r.append(end_time-start_time)
shape.append(X.shape)

start_time = time.time()
pca1 = PCA(n_components=2)
X_pca1 = pca1.fit_transform(X)
kmeans2 = KMeans(n_clusters=3)
kmeans2.fit(X_pca1)
plt.figure()
plt.title("Clustering for database with 2 dimension")
plt.scatter(X_pca1[:, 0], X_pca1[:, 1], c=kmeans2.labels_)
plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[
            :, 1], s=200, marker='*', c='red')
end_time = time.time()
time_r.append(end_time-start_time)
shape.append(X_pca1.shape)

start_time = time.time()
pca2 = PCA(n_components=3)
X_pca2 = pca2.fit_transform(X)
kmeans3 = KMeans(n_clusters=3)
kmeans3.fit(X_pca2)
fig = plt.figure(figsize=(8, 6))
plt.title("Clustering for database with 3 dimension")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca2[:, 0], X_pca2[:, 1], X_pca2[:, 2], c=kmeans3.labels_)
ax.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[
           :, 1], kmeans3.cluster_centers_[:, 2], s=200, marker='*', c='red')
end_time = time.time()
time_r.append(end_time-start_time)
shape.append(X_pca2.shape)

print(time_r)
print(shape)

plt.show()
