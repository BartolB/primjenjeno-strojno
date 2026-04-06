import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("tiger.png")

# povećanje svjetline slike
bright = img + 0.2
bright = np.clip(bright, 0, 1)   

plt.imshow(bright)
plt.title("Povećana svjetlina")
plt.show()


rot = np.rot90(img)

plt.imshow(rot)
plt.title("Rotirana slika")
plt.show()



small = img[::10, ::10]

plt.imshow(small)
plt.title("Smanjena rezolucija")
plt.show()


# prikaz samo druge četvrtine slike po širini
h, w, c = img.shape

result = np.zeros_like(img)      
result[:, w//4:w//2] = img[:, w//4:w//2]

plt.imshow(result)
plt.title("Druga četvrtina slike")
plt.show()