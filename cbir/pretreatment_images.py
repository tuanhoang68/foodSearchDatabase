import os
import glob
import cbir.separate_background as sb
import cbir.featuresVetor as featuresVetor
import pickle
import cbir.kdTree as kdTree
import cbir.base as base


path_image_original = 'CBIR/storage/Image/*.jpg'
path_storage_image  = 'CBIR/storage/Image'
path_tree           = 'CBIR/storage/tree.pkl'
path_separated      = 'CBIR/storage/Image/*_Separate.jpg'
dir_image           = 'CBIR/tests/image_test'


def count_total_images_in_folder(root_dir):
    count = 0
    for dirName, subdirList, fileList in os.walk(root_dir):
        for fname in fileList:
            if fname.endswith('.jpg'):
                count += 1
                
    return count
                


def Create_Separate_Background(root_dir):
    count = 0
    total_img = count_total_images_in_folder(root_dir)
    
    for dirName, subdirList, fileList in os.walk(root_dir):
        # print("subdirList: ", subdirList)
        for fname in fileList:
            if fname.endswith('.jpg'):
                count += 1
                full_path = os.path.join(dirName, fname)
                # print("name: ", fname) # in ra tên của từng tệp
                print("Đang xử lý: ", full_path, " - ", int(round(count/total_img*100)), "%")
                sb.Separate_Background(full_path)
                       
        print("---")


def Get_List_Feature_Vector(root_dir):
    print("::::: Thực hiện trích xuất vector đặc trưng :::::")
    count = 0
    total_img = count_total_images_in_folder(root_dir)
    
    list_vector_path_distance_mapping = []
    
    for dirName, subdirList, fileList in os.walk(root_dir):
        print("subdirList: ", subdirList)
        
        for fname in fileList:
            if fname.endswith('_Separate.jpg'):
                count += 1
                full_path = os.path.join(dirName, fname)
                # print("name: ", fname) # in ra tên của từng tệp
                # print("path: ", full_path)
                feature_vector = featuresVetor.getFeatureVector(full_path)
                
                
                tong = sum(feature_vector)
                print("Tổng vector: ", tong)
                
                
                vector_path_distance_mapping = {
                    "feature_vector"    : feature_vector,
                    "path_img"          : full_path,
                    "distance_to_query" : None
                }
                
                list_vector_path_distance_mapping.append(vector_path_distance_mapping)
                
                if count == total_img:
                    print("Đang xử lý: ", int(round(count/total_img*100)), "%")
                elif count % 3 == 0 : # 3 ảnh in một lần
                    print("Đang xử lý: ", int(round(count/total_img*100)), "%")
        print("---")
    
    print("Hoàn thành!\n")
    return list_vector_path_distance_mapping


def resize_image_separeted_to_512x512(root_dir):
    for dirName,subdirList, fileList in os.walk(root_dir):
        for fname in fileList:
            if fname.endswith('_Separate.jpg'):
                full_path = os.path.join(dirName, fname)
                base.resizeIMG(img_path = full_path)
    
        

# def Get_List_Feature_Vector(path_image_separeted):
#     print("::::: Thực hiện trích xuất vector đặc trưng :::::")
    
#     paths = enumerate(glob.iglob(path_image_separeted))
#     list_vector_path_distance_mapping = []

#     total_path = len(glob.glob(path_image_separeted))
    
#     count = 0
#     for (i, path) in paths:
#         count += 1
        
#         feature_vector = featuresVetor.getFeatureVector(path)
#         vector_path_distance_mapping = {
#             "feature_vector"    : feature_vector,
#             "path_img"          : path,
#             "distance_to_query" : None
#         }
#         list_vector_path_distance_mapping.append(vector_path_distance_mapping)
        
#         # Hiển thị % hoàn thành công việc
#         if count == total_path:
#             print("Đang xử lý: ", int(round(count/total_path*100)), "%")
#         elif count % 3 == 0 : # 3 ảnh in một lần
#             print("Đang xử lý: ", int(round(count/total_path*100)), "%")
        
#     print("Hoàn thành!\n")
#     return list_vector_path_distance_mapping


def Get_Feature_Vectors(path):
    path_image_separeted = enumerate(glob.iglob(path))
    list_feature_vector = Get_List_Feature_Vector(path_image_separeted)
    
    return list_feature_vector


def SaveTree(tree):
    with open(path_tree, "wb") as f:
        pickle.dump(tree, f)


def LoadTree(path_tree):
    with open(path_tree, "rb") as f:
        tree = pickle.load(f)

    return tree


def remove_image_separeted(root_dir):
    for dirName, subdirList, fileList in os.walk(root_dir):
        for fname in fileList:
            if fname.endswith('_Separate.jpg'):
                full_path = os.path.join(dirName, fname)
                os.remove(full_path)
                
                
def main():
    # Xóa các lưu trữ cũ
    remove_image_separeted(path_storage_image)
    if os.path.exists(path_tree):
        os.remove(path_tree)
    
    
    # Tách nền tất cả ảnh
    Create_Separate_Background(path_storage_image)
    
    resize_image_separeted_to_512x512(path_storage_image)
    
    # Trích xuất đặc trưng của tất cả ảnh đã được tách nền
    list_feature_vector = Get_List_Feature_Vector(path_storage_image)

    # # Tạo ra cây k-d
    print("::::: Xây dựng cây K-D :::::")
    tree = kdTree.build_kdtree(list_feature_vector)
    
    # Save tree to file
    SaveTree(tree)
    
    print("Tiền xử lý xong!\nSẵn sàng để truy vấn!\n")
    


if __name__ == '__main__':
    main()
