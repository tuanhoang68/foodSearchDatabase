from scipy.spatial import KDTree

vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
tree = KDTree(vectors)
distances, indices = tree.query([2, 3, 4])

print(distances, indices)