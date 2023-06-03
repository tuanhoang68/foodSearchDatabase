import cbir.pretreatment_images as pi
import cbir.cbir as cbir

def fetchImage(name):
    return 'CBIR/tests/image_test/' + name

def switch_case(argument):
    if argument == "1":
        return pi.main()
    elif argument == "2":
        print('Ảnh muốn truy vấn: ')
        photo = input()
        
        # nạp ảnh
        path_image_query  = fetchImage(photo)
        return cbir.cbir(path_image_query)
    else:
        print("Không hợp lệ! Nhập lại!")
        pass

#  Run
def main():
    while True:
        user_input = input("Service: \n\n[1] Thực hiện tiền xử lý ảnh \n[2] Thực hiện truy xuất ảnh \n[x] Thoát \n\nLựa chọn:  ")
        if user_input == "x":
            break
        print(switch_case(user_input))
        

if __name__ == '__main__':
    main()