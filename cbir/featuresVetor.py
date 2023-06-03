from numpy import concatenate
import cbir.feature.texture.texture as texture
import cbir.feature.color.histogram as histogram
import cbir.feature.color.colorCode as colorCode
import numpy as np


def getFeatureVector(image_path):
    
    color_intensity     =   histogram.Color_Histogram_Intensity(image_path)
    color_code          =   colorCode.getColorCode(image_path)
    texture_feature     =   texture.getTextureMatrix(image_path)

    # Vector đặc trưng
    feature_vector      =   np.concatenate((color_intensity, color_code, texture_feature))
    
    # print("Vector: ", feature_vector)
    return feature_vector
