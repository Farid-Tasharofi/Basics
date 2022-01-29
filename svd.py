import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


img = Image.open('smpl.png')
grey_IMG = img.convert('LA')
np_IMG = np.array(list(grey_IMG.getdata(band=0)), float)
np_IMG.shape = (grey_IMG.size[1], grey_IMG.size[0])
np_IMG = np.matrix(np_IMG)

U, sigma, V = np.linalg.svd(np_IMG)
print(sigma.size)
reconstimg = np.matrix(U[:, :1]) * np.diag(sigma[:1]) * np.matrix(V[:1, :])
for i in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    plt.show()
