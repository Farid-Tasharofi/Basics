import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('noise1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.ones((5, 5), np.float32)/25
dst = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (5, 5))
gblur = cv2.GaussianBlur(img, (9, 9), 4)
median = cv2.medianBlur(img, 9)
bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)
titles = ['image', '2D Convolution', 'blur',
          'GaussianBlur', 'median', 'bilateralFilter']
images = [img, dst, blur, gblur, median, bilateralFilter]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

_, th1 = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(
    img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(
    img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
_, th4 = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
_, th5 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
_, th6 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
_, th7 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

titles = ['Original Image', 'THRESH_MEAN_C', 'THRESH_GAUSSIAN_C,', 'BINARY',
          'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, th1, th2, th3, th4, th5, th6, th7]

for i in range(8):
    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
