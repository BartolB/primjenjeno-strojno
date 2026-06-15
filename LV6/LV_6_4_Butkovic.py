import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image = mpimg.imread('example_grayscale.png')

X = image.reshape(-1, 1)

kmeans = KMeans(n_clusters=10)
kmeans.fit(X)

values = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_

compressed = values[labels]
compressed = compressed.reshape(image.shape)

plt.figure()
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.figure()
plt.imshow(compressed, cmap='gray')
plt.title("Kvantizirana")
plt.show()