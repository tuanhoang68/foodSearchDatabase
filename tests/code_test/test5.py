import numpy as np
import skimage
from skimage import color, img_as_ubyte
import matplotlib.image as mpimg

def normalizationVector (vector):
    return (vector - np.min(vector)) / (np.max(vector) - np.min(vector))


def RGB_to_Gray(image_path): # Chuyển từ ảnh màu về ảnh xám
    # Khởi tạo một ảnh xám từ một ảnh màu. Đầu tiên, nó đọc ảnh màu từ đường dẫn image_path bằng 
    # cách sử dụng hàm mpimg.imread(). 
    img = mpimg.imread(image_path)
    
    # Sau đó, nó tạo ra một mảng grayImage với kích thước bằng với kích thước 
    # của ảnh màu. Mảng này sẽ được sử dụng để lưu trữ ảnh xám.
    gray_image = np.zeros(img.shape)
    
    # Tiếp theo, hàm trích xuất các kênh màu đỏ (Red), xanh lá cây(Green) và xanh dương(Blue) của ảnh màu 
    # và lưu trữ chúng trong các mảng numpy riêng biệt R, G và B
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    
    # Các mảng này được sử dụng để chuyển đổi các giá trị RGB thành thang độ xám bằng 
    # công thức grey = 0,299*R + 0,587*G + 0,114*B.
    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    # Sau đó, hàm tính toán giá trị trung bình của ba giá trị
    # thang độ xám và đặt mảng grayImage thành giá trị này cho mỗi pixel. 
    gray_matrix     =   (R+G+B)
    gray_image      =   img.copy()

    for i in range(3):
        gray_image[:, :, i] = gray_matrix
        
    
    result = {
        "gray_matrix" : gray_matrix,
        "gray_image"  : gray_image
    }
    
    # Cuối cùng, mảng grayImage chứa phiên bản ảnh xám của ảnh màu và được trả về
    return result


def GetCoMatrix(img_zip):
    gray = color.rgb2gray(img_zip)
    
    # Chuyển đổi hình ảnh thang độ xám thành tập các số nguyên không dấu 8 bit.
    image = img_as_ubyte(gray)
    
    matrix_coocurrence = skimage.feature.graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=False, symmetric=False)
    
    # 6. Ma trận đồng xuất hiện được trả về từ hàm.
    return matrix_coocurrence


# GLCM properties

def contrast_feature(matrix_coocurrence):
    # Hàm này lấy ma trận cùng xuất hiện làm đầu vào và trả về đặc trưng 'contrast' (tương phản). 
    # Dưới đây là giải thích từng bước về chức năng của code:

    # 1. Hàm skiage.feature.graycoprops được sử dụng để tính toán đặc trưng contrast của ma trận đồng 
    # xuất hiện đầu vào. Mảng matrix_coocurrence được chuyển làm hình ảnh đầu vào và 'độ tương phản' được 
    # chuyển làm thuộc tính để tính toán.
    contrast = skimage.feature.graycoprops(matrix_coocurrence, 'contrast')
    
    # 2. Kết quả đặc trưng contrast được trả về từ hàm.
    return contrast

    # Tóm lại, hàm này lấy một ma trận đồng xuất hiện làm đầu vào và tính toán đặc trưng contrast của ma trận. 
    # contrast_feature có thể được sử dụng làm đặc trưng kết cấu cho các tác vụ phân tích hình ảnh.
    
    
def energy_feature(matrix_coocurrence):
    energy = skimage.feature.graycoprops(matrix_coocurrence, 'energy')
    return energy

def dissimilarity_feature(matrix_coocurrence):
    energy = skimage.feature.graycoprops(matrix_coocurrence, 'dissimilarity')
    return energy

def homogeneity_feature(matrix_coocurrence):
    energy = skimage.feature.graycoprops(matrix_coocurrence, 'homogeneity')
    return energy

def correlation_feature(matrix_coocurrence):
    energy = skimage.feature.graycoprops(matrix_coocurrence, 'correlation')
    return energy


def getTextureMatrix(image_path):
    # Hàm getTextureMatrix() nhận đầu vào là đường dẫn của một tệp ảnh. Đầu tiên, nó gọi hàm RGB_to_Gray() để 
    # chuyển đổi ảnh màu sang ảnh xám. Sau đó, nó tính toán ma trận lược đồ màu của ảnh xám bằng cách sử dụng
    # hàm GetCoMatrix(). Tiếp theo, nó tính toán hai đặc trưng của ma trận lược đồ màu đó bằng cách sử dụng 
    # hàm energy_feature() và contrast_feature(). Cuối cùng, nó ghép hai đặc trưng này thành một vector đặc 
    # trưng và trả về vector đó.
    
    # Đầu tiên, nó gọi hàm RGB_to_Gray() để chuyển đổi ảnh màu sang ảnh xám.
    img_zip = RGB_to_Gray(image_path)
    
    # Sau đó, nó chuyển ảnh xám thành ma trận lược đồ màu của ảnh xám bằng cách sử dụng hàm GetCoMatrix().
    matrix_coocurrence = GetCoMatrix(img_zip['gray_image'])

    # Tiếp theo, nó tính toán hai đặc trưng của ma trận lược đồ màu đó bằng cách sử dụng hàm energy_feature() 
    # và contrast_feature().
    energy          =   energy_feature(matrix_coocurrence)
    contrast        =   contrast_feature(matrix_coocurrence)
    dissimilarity   =   dissimilarity_feature(matrix_coocurrence)
    homogeneity     =   homogeneity_feature(matrix_coocurrence)
    correlation     =   contrast_feature(matrix_coocurrence)
    
    texture_vector = np.concatenate((energy[0], contrast[0], dissimilarity[0], homogeneity[0], correlation[0]))
    
    # Chuẩn hóa các vector về khoảng [0,1]
    return texture_vector


def main():
    image_path = "C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/Code/CBIR/tests/image_test/ff.jpg"
    print(getTextureMatrix(image_path))
    # input("Please Enter to Continue...")


if __name__ == '__main__':
    main()
