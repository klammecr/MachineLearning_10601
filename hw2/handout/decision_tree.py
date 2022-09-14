# Third Party
from enum import unique
from multiprocessing.sharedctypes import Value
import sys
from math import log2
import numpy as np

def calc_entropy(labels):
    """
    Calculate the entropy of the group of data.
    Format: [x0,..., xn, y]
    """
    entropy     = 0
    unique_lbls = np.unique(labels)
    total_lbls  = len(labels)
    for lbl in unique_lbls:
        # Calculate the contribution then update the entropy
        val_percentage = len(np.where(labels == lbl)) / total_lbls
        entropy       -= val_percentage * log2(val_percentage)

    return entropy

def calc_mi(data, attr):
    """
    Calculate the mutual information for a certain attribute

    :param data: The data left at the current root node
    :param attr: Which attribute to split the data on
    """
    # Entropy at current level - weighted sum of split entropies
    pass

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree      = None

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

    # Create a decision tree object
    dt = DecisionTree(max_depth)
