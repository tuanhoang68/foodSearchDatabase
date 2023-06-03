import numpy as np
import matplotlib.image as mpimg
import cbir.feature.base as base

def RGB2ColorCode(array):
    # Hàm RGB2ColorCode nhận vào một mảng gồm ba số nguyên biểu thị 
    # các giá trị màu đỏ, lục và lam của một màu. 
    
    # Hàm chuyển đổi các giá trị này thành biểu diễn nhị phân của chúng bằng cách sử dụng cú pháp định 
    # dạng chuỗi f và nối hai bit đầu tiên của mỗi chuỗi nhị phân để tạo thành mã nhị phân sáu bit. 
    # Mã này sau đó được trả về dưới dạng một chuỗi.
    
    bit_red     =   f'{array[0]:08b}'
    bit_green   =   f'{array[1]:08b}'
    bit_blue    =   f'{array[2]:08b}'

    zipcode     =   bit_red[0] + bit_red[1] + \
                    bit_green[0] + bit_green[1] + \
                    bit_blue[0] + bit_blue[1]
    # Ví dụ: Đầu vào là mảng [255, 0, 0]
    # >>> RGB2ColorCode([255, 0, 0])
    # '111000' 
    
    # Trong ví dụ này, giá trị màu đỏ là 255, ở dạng nhị phân là 11111111. Hai bit đầu tiên của chuỗi nhị 
    # phân này là 11. Giá trị màu lục là 0, ở dạng nhị phân là 00000000. Hai bit đầu tiên của chuỗi nhị 
    # phân này cũng là 00 . Giá trị màu xanh lam cũng là 0, vì vậy hai bit đầu tiên của nó cũng là 00. 
    # Mã nhị phân sáu bit thu được là 110000, được hàm trả về dưới dạng một chuỗi.

    # Nhìn chung, hàm này nhận giá trị màu RGB và trả về mã nhị phân sáu bit đại diện cho màu.
    return zipcode


# Chuyển nhị phân thành int 
def Binary2Integer(string_binary):
    number_integer = int(string_binary, 2)
    
    # >>> Binary2Integer('1010')
    # 10
    return number_integer


def getColorCode(image_path):
    # Định nghĩa một hàm gọi là getColorCode lấy đường dẫn hình ảnh làm đầu vào. 
    # Hàm đọc hình ảnh bằng hàm mpimg.imread và lấy chiều cao và chiều rộng của hình ảnh bằng thuộc
    # tính hình dạng của mảng hình ảnh.
    
    img             =   mpimg.imread(image_path)
    Image_Height    =   img.shape[0]
    Image_Width     =   img.shape[1]

    bins = np.zeros(64, dtype=int)
    # Sau đó, hàm khởi tạo một mảng NumPy có tên là bins với 64 số không. 
    # Mảng này sẽ được sử dụng để lưu trữ tần suất xuất hiện của từng màu trong ảnh.
    
    
    # Sau đó, hàm lặp qua từng pixel trong ảnh bằng cách sử dụng hai vòng lặp for lồng nhau. 
    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            
            # Đối với mỗi pixel, hàm trích xuất các giá trị màu đỏ, lục và lam của pixel và lưu trữ 
            # chúng trong một mảng có tên là ar. 
            ar = [img[x, y, 0], img[x, y, 1], img[x, y, 2]]
            
            # Sau đó, hàm gọi hàm RGB2ColorCode để chuyển đổi các giá trị RGB thành mã nhị phân sáu bit
            bit_color_6 = RGB2ColorCode(ar)
            
            # Sau đó gọi hàm Binary2Integer để chuyển đổi mã nhị phân thành chỉ số số nguyên. 
            index = Binary2Integer(bit_color_6)
            
            # Số tần số của màu tại chỉ mục này sau đó được tăng lên trong mảng bins.
            bins[index] += 1

    # Sau khi đếm tần số của từng màu, hàm sẽ tính tổng của tất cả các tần số được tính trong mảng bins.
    # sum = np.sum(bins)
    # if sum == 0: # Nếu tổng bằng 0, nó được đặt thành 1 để tránh lỗi chia cho 0. Sau đó, hàm chia từng phần tử trong 
    #     sum = 1  # mảng bins cho tổng để chuẩn hóa tần số đếm thành xác suất. Mảng xác suất kết quả sau đó được hàm trả về.
    # bins = np.true_divide(bins, sum)  # <=> bins /= sum
    
    # Nhìn chung, hàm này nhận một đường dẫn hình ảnh, đọc hình ảnh, đếm tần suất xuất hiện của từng màu
    # trong hình ảnh và trả về một mảng xác suất thể hiện sự phân bố màu sắc trong hình ảnh.
    
    # Dưới đây là một ví dụ sử dụng chức năng:
    # >>> getColorCode('path/to/image.png')
    # array([0.1, 0. , 0. , 0. , 0.2, 0. , 0. , 0. , ... , 0. , 0. , 0. ])
    
    return base.normalizationVector(bins)


def main():
    image_path = "ff.jpg"
    print(getColorCode(image_path))
    # input("Please Enter to Continue...")


if __name__ == '__main__':
    main()
