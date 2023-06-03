import pickle

def LoadTree(path_tyree):
    with open(path_tyree, "rb") as f:
        ttree = pickle.load(f)
    return ttree
    # return 1

def print_kdtree(node, level=0):
    if node is not None:
        print(' ' * level, node.point)
        print_kdtree(node.left, level + 1)
        print_kdtree(node.right, level + 1)
        
        
path_tree   =   'C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/Code/CBIR/storage/tree.pkl'
tree = LoadTree(path_tree)
print_kdtree(tree)