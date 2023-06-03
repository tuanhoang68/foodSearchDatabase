import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import os

import cbir.kdTree as kdTree
import cbir.featuresVetor as featuresVetor
import cbir.separate_background as sb
import cbir.pretreatment_images as pi

length_feature_vector   =   484 # Độ dài vector đặc trưng
quantity_img_display    =   10  # Số lượng ảnh cần hiển thị
path_tree               =   'CBIR/storage/tree.pkl'
temp_path               =   'CBIR/tests/temp/temp.jpg'




def order_nearest_imgs (neighbors):
    # Sắp xếp lại theo khoảng cách
    neighbors_ordered = sorted(neighbors, key=lambda point: point['distance_to_query'])
    return neighbors_ordered
    
    
    
def get_display(neighbors):
    axes = []
    grid_size = int(math.sqrt(quantity_img_display))
    
    # Tạo một lưới ảnh fig với kích thước 10x5
    fig = plt.figure(figsize=(10, 5))

    for id in range(quantity_img_display):
        # Lấy ra cặp giá trị từ danh sách "vector_distance_mapping"
        draw_image = neighbors[id]
        
        # Thêm một đối tượng Axes vào danh sách "axes".
        axes.append(fig.add_subplot(grid_size, grid_size + 1, id+1))

        #  Set vị trí & khoảng cách so sánh của từng item
        item_name = "[Ảnh " + str(id) +"]\n" + str(draw_image['distance_to_query'])
        axes[-1].set_title(item_name)
        
        file_name = draw_image['path_img'].replace("_Separate.jpg", ".jpg")
        
        plt.imshow(Image.open(file_name))

    # Tự động điều chỉnh kích thước của các đối tượng trong Figure.
    fig.tight_layout()
    
    # Đặt tên cho ảnh output
    result_path = "Result.jpg"  # 
    plt.savefig(result_path)    #  Lưu hình ảnh

    # Trả về một dictionary chứa đường dẫn tới file ảnh lưới được lưu lại, 
    # các index tương ứng với các vector đặc trưng gần nhất và tên file 
    # ảnh lưới.
    # match_images_package = dict()
    match_images_package = {
        "path" : result_path  # Đường dẫn ảnh output
    }         

    return match_images_package  # Trả về 1 object



def Process_Image_Query(path_image_query):
    path_image_query_Separated  = sb.Create_Separate_Background(path_image_query)
    
    import cbir.base as base
    _ = base.resizeIMG(img_path = path_image_query_Separated)
    
    feature_vector_img_query    = featuresVetor.getFeatureVector(path_image_query_Separated)

    return feature_vector_img_query



def print_kdtree(node, level=0):
    if node is not None:
        print(' ' * level, node.point)
        print_kdtree(node.left, level + 1)
        print_kdtree(node.right, level + 1)



def cbir(path_image_query):
    # Lấy vector đặc trưng của ảnh truy vấn
    feature_vector_img_query = Process_Image_Query(path_image_query)
    
    tong = sum(feature_vector_img_query)
    print("Tổng vector: ", tong)
    # Nạp cây K-D, nếu tồn tại thì xóa cây
    tree = None
    tree = pi.LoadTree(path_tree)
    
    print_kdtree(tree)
    
    # Tìm 10 neighbors gần nhất
    neighbors = kdTree.k_nearest_neighbors(tree, feature_vector_img_query, k = quantity_img_display)
    # print("\n10 nearest neighbors: ", neighbors)
    
    neighbors_ordered = order_nearest_imgs(neighbors)

    result = get_display(neighbors_ordered)

    # Display
    os.startfile(result["path"])
    
    # result = relevanceFeedback(neighbors_ordered, feature_vector_img_query, tree)
    
    # os.startfile(result["path"])
    
    print('Nhấn phím bất kỳ để tiếp tục!')
    photo = input()
    ##############

    # Delete image in folder
    os.remove(result["path"])
    # os.remove(temp_path)




def relevanceFeedback(neighbors_ordered, feature_vector_img_query, tree):
    # Kỹ thuật RELEVANCE FEEDBACK (phản hổi tương đồng)
    # Thuật toán Rocchio
    print('Chọn một số bức ảnh bạn cảm thấy tương tự:')
    print('Vị trí ảnh: ')
    photo = input()

    # os.remove(result["path"])

    marker_array    =   [False for _ in range(quantity_img_display)]  # Khởi tạo mảng đánh dấu ban đầu toàn False
    count_relevant  =   0
    
    # Tách input thành các số, sau đó sửa lại mảng đánh dấu
    for t in photo.split():
        try:
            marker_array[int(t)] = True
            count_relevant += 1
            
        except ValueError:
            print('Hãy nhập số!')
            pass
        
    # print(len(marker_array))

    relevant            =   np.zeros(length_feature_vector)
    non_relevant        =   np.zeros(length_feature_vector)
    count_non_relevant  =   quantity_img_display - count_relevant
    
    
    for index, value in enumerate(marker_array): # enumerate(marker_array) biến marker_array thành các cặp index, value
        if value == False:
            # Cộng vector không phải match_image vào vector non_relevant, để tí nữa tính công thức RELEVANCE FEEDBACK
            for i in range(0, length_feature_vector):
                non_relevant[i] += neighbors_ordered[index]['feature_vector'][i]
        else:
            # Cộng vector match_image vào vector relevant, để tí nữa tính công thức RELEVANCE FEEDBACK
            for i in range(0, length_feature_vector):
                relevant[i] += neighbors_ordered[index]['feature_vector'][i]
    
    
    
    # Hệ số của thuật toán
    alpha   =   1
    beta    =   0.75
    gamma   =   0.25

    vector_RF = alpha*feature_vector_img_query  +  (beta/count_relevant)*relevant  -  (gamma/count_non_relevant)*non_relevant
    # print(vector_RF)

    # Tính toán lại các neighbors với vector mới là vector_RF
    neighbors = kdTree.k_nearest_neighbors(tree, vector_RF, k=quantity_img_display)
    neighbors_ordered = order_nearest_imgs(neighbors)

    return get_display(neighbors_ordered)
