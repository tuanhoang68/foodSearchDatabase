import numpy as np

def normalizationVector (vector):
    return (vector - np.min(vector)) / (np.max(vector) - np.min(vector))

