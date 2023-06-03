import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
from skimage import io, color, img_as_ubyte
from scipy import misc
import imageio.v2 as imageio


def Array2Binary(array):
    bit_r = f'{array[0]:08b}'
    bit_g = f'{array[1]:08b}'
    bit_b = f'{array[2]:08b}'

    bit_rgb_zipcode = bit_r[0] + bit_r[1] + \
        bit_g[0] + bit_g[1] + bit_b[0] + bit_b[1]
    return bit_rgb_zipcode


def Binary2Integer(string_binary):
    number_integer = int(string_binary, 2)
    return number_integer


def RGB_to_Gray(img):
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


img = io.imread('CBIR/begin/ff.jpg')
gray = color.rgb2gray(img)
image = img_as_ubyte(gray)
io.imshow('CBIR/begin/ff.jpg')
io.show()

bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144,
                160, 176, 192, 208, 224, 240, 255])  # 16-bit
inds = np.digitize(image, bins)

max_value = inds.max()+1
matrix_coocurrence = skimage.feature.graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                                  levels=max_value, normed=False, symmetric=False)

# GLCM properties


def contrast_feature(matrix_coocurrence):
    contrast = skimage.feature.graycoprops(matrix_coocurrence, 'contrast')
    return "Contrast = ", contrast


def energy_feature(matrix_coocurrence):
    energy = skimage.feature.graycoprops(matrix_coocurrence, 'energy')
    return "Energy = ", energy


def entropy_feature(matrix_coocurrence):
    entropy = skimage.feature.graycoprops(matrix_coocurrence, 'entropy')
    return "Entropy = ", entropy


print(contrast_feature(matrix_coocurrence))
print(energy_feature(matrix_coocurrence))
