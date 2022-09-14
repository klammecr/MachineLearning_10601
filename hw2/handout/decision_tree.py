# Third Party
import sys
import numpy as np

# In House
from inspection import calc_entropy
from majority_vote import DataInterface

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
        self.max_depth = max_depth
        self.tree      = None

    def fit(self, X, y, hdrs):
        # Base Case:
        if data is None:
            return self.tree
        else:
            attr_mi = {}
            # Find the mutual information for each attribute
            for hdr in hdrs:
                attr_mi[hdr] = calc_mi(X, y, hdrs, hdr)
            
            # Find out which feature had the highest mutual information
            max_value, idx = max(attr_mi.values)

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

    # Create a decision tree object
    dt = DecisionTree(max_depth)
    dt.fit(X, y, hdrs)
