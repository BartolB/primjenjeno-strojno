import numpy as np
import matplotlib.pyplot as plt

def sahovnica(size):
    
    # crno i bijelo polje
    black = np.zeros((size, size))
    white = np.ones((size, size)) * 255
    
    # jedan red (crno-bijelo-crno-bijelo...)
    row1 = np.hstack((black, white, black, white))
    
    # drugi red
    row2 = np.hstack((white, black, white, black))
    
    # složiti cijelu šahovnicu
    board = np.vstack((row1, row2, row1, row2))
    
    return board


img = sahovnica(50)

plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
