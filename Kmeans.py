import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

y=load_digits()
x = y.data
k = 10
n, d = x.shape
centroids = x[np.random.choice(n, k, replace=False)]
obj_vals = []

max_iter = 100

for _ in range(max_iter):
    distances = np.linalg.norm(x[:, np.newaxis, :] - centroids, axis=-1)
    labels = np.argmin(distances, axis = 1 )
    obj_val = np.sum(np.linalg.norm(x - centroids[labels], axis=1)**2)
    obj_vals.append(obj_val)
    for j in range(k):
        centroids[j] = np.mean(x[labels == j], axis = 0)

plt.figure()
plt.plot(range(1, max_iter+1), obj_vals)
plt.xlabel('Iteration number')
plt.ylabel('Objective value')
plt.title('K-means objective value vs. iteration number')

plt.figure()
for i in range(k):
    plt.subplot(2, 5, i+1)
    plt.imshow(centroids[i].reshape(8, 8), cmap='gray')
    plt.axis('off')
plt.suptitle('Cluster centers')
plt.show()