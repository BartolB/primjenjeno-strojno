import numpy as np
import matplotlib.pyplot as plt

# koordinate
x = np.array([1, 2, 3, 3, 1])
y = np.array([1, 2, 2, 1, 1])

plt.figure()
plt.plot(x, y, marker='o', color='blue', linewidth=2)

plt.title("Primjer")
plt.xlabel("X os")
plt.ylabel("Y os")

plt.xlim(0,4)
plt.ylim(0,4)

plt.show()
