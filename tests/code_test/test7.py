from PIL import Image
import os

root_dir = "C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/dataset/archive/images/apple_pie_1"
temp_path = "C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/dataset/archive/images"

def resize_image_separeted_to_512x512(root_dir):
    for dirName,subdirList, fileList in os.walk(root_dir):
        for fname in fileList:
            if fname.endswith('.jpg'):
                full_path = os.path.join(dirName, fname)
                img = Image.open(full_path)
                width, height = img.size
                img.close() # đóng file ảnh
                if height != 512 or width != 512:
                    # print(f'Kích thước của ảnh là {width}x{height}')
                    os.remove(full_path)
                
                

# resize_image_separeted_to_512x512(root_dir)
resize_image_separeted_to_512x512(temp_path)