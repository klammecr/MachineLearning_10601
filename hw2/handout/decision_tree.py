# Third Party
from asyncore import write
from posixpath import split
import sys
import numpy as np

# In House
from inspection import calc_entropy
from majority_vote import DataInterface, MajorityVoteClassifier, calc_error, write_error

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
        self.mvc       = MajorityVoteClassifier()
        self.tree      = None

    def fit(self, X, y, hdrs):
        self.tree = self.recurse_tree(X, y, hdrs)
        return self.tree

    def predict_batch(self, node, X_test, test_hdrs):
        preds = []
        for sample in X_test:
            sample_dict = create_sample_dict(sample, test_hdrs)
            preds.append(self.predict(tree, sample_dict))
        return preds

    def predict(self, node, X_dict):
        if node is None:
            return None

        # Return value
        ret = None

        # Base case, node is a leaf node
        if node.vote is not None:
            ret =  node.vote
        # Recursive Step:
        else:
            if node.attr in X_dict.keys():
                attr_val = X_dict[node.attr]
                if attr_val == 0:
                    ret = self.predict(node.right, X_dict)
                else:
                    ret = self.predict(node.left, X_dict)
        return ret

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
            curr.attr = hdrs[split_idx] # Mark the split attribute for the current node

            # Mark the headers that are left after the split
            remaining_hdrs = np.r_[hdrs[:split_idx], hdrs[split_idx+1:]]

            # Take each value for this feature (0 or 1)
            # Left child always corresponds to 1, Right child always corresponds to 0
            left_idxs  = X[:, split_idx] == 1
            right_idxs = X[:, split_idx] == 0

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

def create_sample_dict(sample, hdrs):
    out = {}
    for idx, hdr in enumerate(hdrs):
        out[hdr] = sample[idx]
    return out

def write_preds(file, preds):
    # Write the test results to a file
    with open(file, "w+") as f:
        for pred in preds:
            f.write(f"{pred}\n")

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
    train_data = data_interface.get_data()
    hdrs = data_interface.get_headers()
    X = train_data[:, :-1]
    y = train_data[:, -1]
    hdrs = hdrs[:-1]


    # Create a decision tree object
    dt = DecisionTree(max_depth)
    tree = dt.fit(X, y, hdrs)

    # Write the predictions of the training set
    train_preds = dt.predict_batch(tree, train_data, hdrs)
    write_preds(train_out, train_preds)

    # Read the test data
    data_interface.read_tsv(test_input)
    test_data  = data_interface.get_data()
    test_hdrs  = data_interface.get_headers()

    # Write the predictions of the test set
    test_preds = dt.predict_batch(tree, test_data, test_hdrs)
    write_preds(test_out, test_preds)

    # Write the metrics
    write_error(train_out, y, test_out, test_data[:, -1], metrics_out)