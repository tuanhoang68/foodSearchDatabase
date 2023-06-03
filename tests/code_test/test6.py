import numpy as np
import cv2

# đọc ảnh đầu vào
image_path = "C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/Code/CBIR/tests/image_test/ff.jpg"
img = cv2.imread(image_path)

# chuyển đổi ảnh sang grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# tính toán gradient của ảnh
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
gradient_magnitude, gradient_angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)

# chia ảnh thành các ô và tính toán histogram các hướng gradient trong mỗi ô
cell_size = (8, 8)
cell_stride = (8, 8)
num_bins = 9
histograms = []
for i in range(0, gradient_magnitude.shape[0] - cell_size[0] + 1, cell_stride[0]):
    row_histograms = []
    for j in range(0, gradient_magnitude.shape[1] - cell_size[1] + 1, cell_stride[1]):
        cell_magnitude = gradient_magnitude[i:i+cell_size[0], j:j+cell_size[1]]
        cell_angle = gradient_angle[i:i+cell_size[0], j:j+cell_size[1]]
        cell_histogram = np.zeros((num_bins,))
        for k in range(num_bins):
            bin_angle_min = k * (180 / num_bins)
            bin_angle_max = (k + 1) * (180 / num_bins)
            bin_mask = np.logical_and(cell_angle >= bin_angle_min, cell_angle < bin_angle_max)
            cell_histogram[k] = np.sum(cell_magnitude[bin_mask])
        row_histograms.append(cell_histogram)
    histograms.append(row_histograms)

# kết hợp các ô thành các khối và chuẩn hóa các vector đặc trưng của các khối
block_size = (2, 2)
block_stride = (1, 1)
epsilon = 1e-7
features = []
for i in range(0, len(histograms) - block_size[0] + 1, block_stride[0]):
    for j in range(0, len(histograms[0]) - block_size[1] + 1, block_stride[1]):
        block_histograms = [histograms[i+k][j+l] for k in range(block_size[0]) for l in range(block_size[1])]
        block_features = np.concatenate(block_histograms)
        block_features /= np.sqrt(np.sum(block_features ** 2) + epsilon)
        features.append(block_features)

# tạo ra một vector đặc trưng cuối cùng bằng cách ghép các vector đặc trưng của các khối lại với nhau
features = np.concatenate(features)

print("Feature: ", features)
print("Length Feature: ", len(features))