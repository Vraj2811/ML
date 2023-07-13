import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

y=load_digits()
x = y.data
k = 10
n, d = x.shape
centers = x[np.random.choice(n, 1)]

def closest_center(p, centers):
  closest, dclosest = -1, np.inf
  for c in centers:
    d = np.linalg.norm(p - c)
    if(d < dclosest):
      dclosest = d
      closest = c
  return (closest, dclosest)

def far(points, centers):
  dmax = 0
  point=-1
  for p in points:
    c, d = closest_center(p, centers)
    if d>dmax:
      point=p
  return point

for _ in range(k-1):
    new_center = far(x,centers)
    centers = np.vstack((centers, new_center))

distances = np.linalg.norm(x[:, np.newaxis, :] - centers, axis=-1)
labels1 = np.argmin(distances, axis = 1)

kcenter_objective = 0
for i in range(len(labels1)):
  kcenter_objective = max(kcenter_objective, distances[i][labels1[i]])

print("Kcentre objective value is ",kcenter_objective)

centroids = x[np.random.choice(n, k, replace=False)]
for _ in range(100):
    distances = np.linalg.norm(x[:, np.newaxis, :] - centroids, axis=-1)
    labels2 = np.argmin(distances, axis = 1 )
    for j in range(k):
        centroids[j] = np.mean(x[labels2 == j], axis = 0)

cluster_overlap = np.zeros((k, k), dtype=int)
for i in range(k):
    for j in range(k):
        cluster_overlap[i][j] = np.sum(np.logical_and(labels1 == i, labels2 == j))

print("The cluster overlap matrix is:")
print(cluster_overlap)