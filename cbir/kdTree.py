import heapq
import pickle
import random
import warnings
import numpy as np
from scipy.spatial.distance import cityblock

class Node:
    def __init__(self, point, depth=0):
        self.point  =   point
        self.left   =   None
        self.right  =   None
        self.depth  =   depth


def build_kdtree(points, depth=0):
    # Đầu tiên kiểm tra xem danh sách các điểm có rỗng hay không. 
    # Nếu danh sách rỗng, trả về None
    if len(points) <= 0:
        return None

    # Chọn trục dựa trên độ sâu để trục quay vòng qua tất cả các giá trị hợp lệ
    k = len(points[0]['feature_vector'])

    axis = depth % k
    # print("axis: ", axis)

    # Sắp xếp điểm dựa trên trục
    sorted_points = sorted(points, key=lambda point: point['feature_vector'][axis])

    # Build node
    mid         =   len(sorted_points) // 2
    node        =   Node(sorted_points[mid], depth)
    node.left   =   build_kdtree(sorted_points[:mid], depth + 1)
    node.right  =   build_kdtree(sorted_points[mid+1:], depth + 1)

    return node

# Tính khoảng cách
def get_distance(point1, point2):
    # Euclid
    return sum([(point1[i] - point2[i]) ** 2 for i in range(len(point1))])
    
    # # Manhattan
    # return cityblock(point1, point2)


def k_nearest_neighbors(tree, vector_x, k=10):
    queue = [(0, tree)]
    neighbors = []

    while len(queue) > 0 and len(neighbors) < k:
        _, node = heapq.heappop(queue)
        if node is None:
            continue

        distance = get_distance(node.point['feature_vector'], vector_x)
        node.point['distance_to_query'] = distance
        # print("Path image: ", node.point['path_img'])
        # print("Distance: ", distance)
        
        for child in [node.left, node.right]:
            if child is not None:
                heapq.heappush(queue, (get_distance(child.point['feature_vector'], vector_x), child))

        if len(neighbors) < k:
            neighbors.append(node.point)
        else:
            farthest_distance = max([get_distance(n['feature_vector'], vector_x) for n in neighbors])
            if distance < farthest_distance:
                neighbors.remove(max(neighbors, key=lambda n: get_distance(n, vector_x)))
                neighbors.append(node.point)

    return neighbors


import numpy as np

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def find_nearest_neighbor(root, point, depth=0):
    if root is None:
        return None

    k = len(point['feature_vector'])
    axis = depth % k

    if point['feature_vector'][axis] < root.point['feature_vector'][axis]:
        best = find_nearest_neighbor(root.left, point, depth + 1)
        opposite = root.right
    else:
        best = find_nearest_neighbor(root.right, point, depth + 1)
        opposite = root.left

    if best is None or distance(point['feature_vector'], best.point['feature_vector']) > distance(point['feature_vector'], root.point['feature_vector']):
        best = root.point

    if distance(point['feature_vector'], best.point['feature_vector']) > abs(point['feature_vector'][axis] - root.point['feature_vector'][axis]):
        candidate = find_nearest_neighbor(opposite, point, depth + 1)
        if candidate is not None and distance(point['feature_vector'], candidate.point['feature_vector']) < distance(point['feature_vector'], best.point['feature_vector']):
            best = candidate.point

    return best



