import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb_to_gray(img):
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R+G+B)
    grayImage = img.copy()

    for i in range(3):
        grayImage[:, :, i] = Avg

    return grayImage


image = mpimg.imread("ff.jpg")
grayImage = rgb_to_gray(image)
plt.imshow(grayImage)
plt.show()
