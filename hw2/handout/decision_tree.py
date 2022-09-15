# Third Party
from posixpath import split
import sys
import numpy as np

# In House
from inspection import calc_entropy
from majority_vote import DataInterface, MajorityVoteClassifier

def calc_mi(data, lbls, hdrs, attr):
    """
    Calculate the mutual information for a certain attribute

    :param data: The data left at the current root node
    :param attr: Which attribute to split the data on
    """
    if attr not in hdrs:
        return None

    # Entropy at current level - weighted sum of split entropies
    mi = calc_entropy(lbls)

    # Find the column we need
    idx         = attr == hdrs
    feature     = data[:, idx]
    num_samples = len(feature)

    # Find the number of unique values
    unique_vals = np.unique(feature)

    # For each value, subtract the weighted entropy from the entropy before splitting
    for val in unique_vals:
        # Find out how many samples have that value (percentage) and what is the entropy of the labels
        val_mask        = feature == val
        val_percentage  = len(np.extract(val_mask, feature)) / num_samples
        # Percentage of samples with that label * the entropy of the labels where the sample has that value
        mi -= val_percentage * calc_entropy(lbls[val_mask.ravel()])

    return mi

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = int(max_depth)
        self.root      = Node()
        self.mvc       = MajorityVoteClassifier()

    def fit(self, X, y, hdrs):
        return self.recurse_tree(X, y, hdrs)


    def recurse_tree(self, X, y, hdrs, depth = 0):
        # Create a node for the current branch
        curr = Node()

        # Criteria if we are stopping (is a leaf node)
        leaf = (depth >= self.max_depth or 
                X is None or np.prod(X.shape) == 0 or
                y is None or np.prod(y.shape) == 0 or
                len(np.unique(y)) == 1 or
                len(np.unique(X)) == 1)

        # Printing the split of 0's and 1's for this branch of the tree
        print(f"[{len(np.extract(y == 0, y))} 0/{len(np.extract(y == 1, y))} 1]")

        # Base Case (Leaf Node)
        if leaf:
            self.mvc.fit(y)
            curr.vote = self.mvc.infer()
        # Recursive Step (Internal Node)
        else:
            attr_mi = {}

            # Find the mutual information for each attribute
            for hdr in hdrs:
                attr_mi[hdr] = calc_mi(X, y, hdrs, hdr)

            # Find out which feature had the highest mutual information
            split_idx = np.argmax(np.array(list(attr_mi.values())))

            # Take each value for this feature (0 or 1)
            # Left child always corresponds to 1, Right child always corresponds to 0
            left_idxs  = X[:, split_idx] == 1
            right_idxs = X[:, split_idx] == 0

            remaining_hdrs = np.r_[hdrs[:split_idx], hdrs[split_idx+1:]]

            # Recursion and printing for debugging
            pipes = "| " * (depth + 1)
            print(f"{pipes}{hdrs[split_idx]} - 0: ", end="")
            curr.right = self.recurse_tree(np.c_[X[right_idxs, :split_idx], X[right_idxs, split_idx+1:]], y[right_idxs], remaining_hdrs, depth+1)
            print(f"{pipes}{hdrs[split_idx]} - 1: ", end="")
            curr.left = self.recurse_tree(np.c_[X[left_idxs, :split_idx], X[left_idxs, split_idx+1:]], y[left_idxs], remaining_hdrs, depth+1)

        return curr
        
class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None

if __name__ == '__main__':
    
    # Arguments: <train input> <testinput> <max depth> <train out> <test out> <metrics out>.
    if len(sys.argv) == 7:
        train_input = sys.argv[1]
        test_input  = sys.argv[2]
        max_depth   = sys.argv[3]
        train_out   = sys.argv[4]
        test_out    = sys.argv[5]
        metrics_out = sys.argv[6]
    else:
        raise ValueError("python decision_tree.py <train input> <testinput> <max depth> <train out> <test out> <metrics out>")

    # Read the training file
    data_interface = DataInterface()
    data_interface.read_tsv(train_input)
    data = data_interface.get_data()
    hdrs = data_interface.get_headers()
    X = data[:, :-1]
    y = data[:, -1]
    hdrs = hdrs[:-1]

    # Create a decision tree object
    dt = DecisionTree(max_depth)
    tree = dt.fit(X, y, hdrs)
    pass