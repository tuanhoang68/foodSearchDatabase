import numpy as np
from PIL import Image

# Đọc bức ảnh vào dưới dạng một mảng numpy
img = np.array(Image.open('ten-tep-anh.png'))

# Chia mảng thành 64 phần bằng nhau
img_reshaped = img.reshape(64, 64, 64, 64)

# Cắt đi 28 phần rìa ngoài của bức ảnh
img_cropped = img_reshaped[:, :, 28:36, 28:36]