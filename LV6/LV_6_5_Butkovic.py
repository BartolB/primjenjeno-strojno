import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


image = mpimg.imread('example.png')

X = image.reshape(-1, 3)

kmeans = KMeans(n_clusters=10)
kmeans.fit(X)

values = kmeans.cluster_centers_
labels = kmeans.labels_

compressed = values[labels]
compressed = compressed.reshape(image.shape)

plt.figure()
plt.imshow(image)
plt.title("Original")

plt.figure()
plt.imshow(compressed.astype(np.uint8))
plt.title("Kvantizirana")
plt.show()