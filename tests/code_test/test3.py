import os
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

# dir = "C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/Code/CBIR/tests/image_test"
dir = "C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/Code/CBIR/storage/Image/"
def iterate_directory(root_dir):
    for dirName, subdirList, fileList in os.walk(root_dir):
        print("subdirList: ", subdirList)
        print("dirName: ", dirName)
        for fname in fileList:
            # if fname.endswith('.jpeg') or fname.endswith('.jpg') or fname.endswith('.png'):
            # if fname.endswith('_Separate.jpg'):
            if fname.endswith('.jpg'):
                full_path = os.path.join(dirName, fname)
                print("name: ", fname) # in ra tên của từng tệp
                print("path: ", full_path)
        print("--------")
                # extract image features using PySIFT, PyORB or PySURF
                # do something with the features
                
                
def remove_image_separeted(root_dir):
    for dirName, subdirList, fileList in os.walk(root_dir):
        for fname in fileList:
            if fname.endswith('_Separate.jpg'):
                full_path = os.path.join(dirName, fname)
                os.remove(full_path)
                print("0_path: ", full_path)
                
                
iterate_directory(dir)
remove_image_separeted(dir)


# # Iterate over all images in the directory
# for filename in os.listdir(dir):
#     if filename.endswith('.jpg'):
#         # Load the image
#         image = cv2.imread(os.path.join(dir, filename))
#         # Convert the image to grayscale
#         image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         print("\n\n\nimage_gray: ", image_gray[0][0])
#         print("\n\n\nimage_gray: ", len(image_gray[511]))
#         # Calculate the grey-level co-occurrence matrix
#         g = greycomatrix(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        
#         # Calculate statistics on the co-occurrence 
        
#         contrast = greycoprops(g, 'contrast')
#         dissimilarity = greycoprops(g, 'dissimilarity')
#         homogeneity = greycoprops(g, 'homogeneity')
#         energy = greycoprops(g, 'energy')
#         correlation = greycoprops(g, 'correlation')
        
#         # Calculate the probability mass function
#         p = g / np.sum(g)
#         # Calculate the entropy
#         entropy = -np.sum(p * np.log2(p + (p == 0)))
        
        
#         print("contrast: ", contrast)
#         print(contrast[0])
#         print("dissimilarity: ", dissimilarity)
#         print("homogeneity: ", homogeneity)
#         print("energy: ", energy)
#         print("correlation: ", correlation)
#         print('Entropy: ', entropy)