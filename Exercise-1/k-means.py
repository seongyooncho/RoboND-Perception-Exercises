import numpy as np
import cv2
import matplotlib.pyplot as plt
from extra_functions import cluster_gen


# Generate some clusters!
n_clusters = 7
clusters_x, clusters_y = cluster_gen(n_clusters)

# Convert to a single dataset in OpenCV format
data = np.float32((np.concatenate(clusters_x), np.concatenate(clusters_y))).transpose()

# Define k-means parameters
# Number of clusters to define
k_clusters = 7
# Maximum number of iterations to perform
max_iter = 10
# Accuracy criterion for stopping iterations
epsilon = 1.0
# Define criteria in OpenCV format
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
# Call k-means algorithm on your dataset
compactness, label, center = cv2.kmeans(data, k_clusters, None, criteria, 10, 
                                        cv2.KMEANS_RANDOM_CENTERS)

# Define some empty lists to receive k-means cluster points
kmeans_clusters_x = []
kmeans_clusters_y = []

# Extract k-means clusters from output
for idx in range(k_clusters):
  kmeans_clusters_x.append(data[label.ravel()==idx][:,0])
  kmeans_clusters_y.append(data[label.ravel()==idx][:,1])

# Plot up a comparison of original clusters vs. k-means clusters
fig = plt.figure(figsize=(12, 6))
min_x = np.min(data[:, 0])
min_y = np.min(data[:, 1])
max_x = np.max(data[:, 0])
max_y = np.max(data[:, 1])
plt.subplot(121)
for idx, xpts in enumerate(clusters_x):
  ypts = clusters_y[idx]
  plt.plot(xpts, ypts, 'o')
  plt.xlim(min_x, max_x)
  plt.ylim(min_y, max_y)
  plt.title('Original Clusters')
plt.subplot(122)
for idx, xpts in enumerate(kmeans_clusters_x):
  ypts = kmeans_clusters_y[idx]
  plt.plot(xpts, ypts, 'o')
  plt.xlim(min_x, max_x)
  plt.ylim(min_y, max_y)
  plt.title('k-means Clusters')

plt.show()

