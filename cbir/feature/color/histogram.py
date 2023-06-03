import numpy as np
import cbir.feature.texture.texture as texture
import math
import cbir.feature.base as base

quantity_segment = 36


def segmentImage(image, number_of_segments):
    imageSegmentHistogram = []
    
    image_height = image.shape[0]
    image_width  = image.shape[1]

    windowsize_r = int(image_height / int(math.sqrt(number_of_segments)))
    windowsize_c = int(image_width  / int(math.sqrt(number_of_segments)))
    
    for x in range(0, image_height - windowsize_r + 1, windowsize_r):
        for y in range(0, image_width - windowsize_c + 1, windowsize_c):
            histogram = [0 for _ in range(256)]
            for i in range(x, x + windowsize_r):
                for j in range(y, y + windowsize_c):
                    histogram[image[i, j][0]] += 1
            imageSegmentHistogram.append(histogram)

    return imageSegmentHistogram



def Color_Histogram_Intensity(image_path):
    listBins = []
    
    Input_Image = texture.RGB_to_Gray(image_path)
    
    listSegmentHistogram = segmentImage(Input_Image['gray_image'], quantity_segment)
    
    for segment in listSegmentHistogram:
        bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        x = 0
        for bin in bins:
            bins[bin] = 0
            if x < 240:
                k = x + 10
            else:
                k = x + 16
            while x < k:
                if x == 256:
                    break
                bins[bin] += segment[x]
                x += 1

        bins = base.normalizationVector(bins)
        listBins.extend(bins) 
        
    return listBins

