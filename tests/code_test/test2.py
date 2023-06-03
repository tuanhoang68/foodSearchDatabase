import math
import numpy as np
import skimage
from skimage import color, img_as_ubyte
import matplotlib.image as mpimg


def RGB_to_Gray(image_path):
    img = mpimg.imread(image_path)
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


def SegmentImage(image, number_of_segments):
    imageSegment = []
    
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    print(image_height)
    print(image_width)

    windowsize_r = int(image_height / int(math.sqrt(number_of_segments)))
    windowsize_c = int(image_width / int(math.sqrt(number_of_segments)))
    
    print(windowsize_r)
    print(windowsize_c)

    count = 0
    for x in range(0, image_height - windowsize_r + 1, windowsize_r):
        for y in range(0, image_width - windowsize_c + 1, windowsize_c):
            histogram = np.zeros([256], np.int32)
            for i in range(x, x + windowsize_r):
                for j in range(y, y + windowsize_c):
                    histogram[image[i, j]] += 1
            imageSegment.append(histogram)

    print(count)
    print("len histogram: ",len(histogram))
    print("histogram: ", histogram)
    print("imageSegment: ", imageSegment)
    return imageSegment


def Color_Histogram_Intensity(image_path):
    return RGB_to_Gray(image_path)


def main():
    # pathIMG = 'C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/Code/CBIR/tests/image_test/' + "ff_Separate.jpg"
    # image = Color_Histogram_Intensity(pathIMG)
    # SegmentImage(image, 16)
    
    
    # Define two example lists
    list1 = [1, 2, 3, 78]
    list2 = [4, 5, 6]

    # Concatenate the lists using the '+' operator
    concatenated1 = list1 + list2
    print(concatenated1)  # Output: [1, 2, 3, 4, 5, 6]

    # Alternatively, extend list1 with the elements of list2
    list1.extend(list2)
    print(list1)  # Output: [1, 2, 3, 4, 5, 6]
    # from PIL import Image
    # image = Image.open(pathIMG)
    # sunset_resized = image.resize((512, 512))
    # sunset_resized.save('C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/Code/CBIR/tests/image_test/' + "ff_Separate2.jpg")


if __name__ == '__main__':
    main()
