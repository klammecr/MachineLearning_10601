
# Third Party
import sys
from math import log2
import numpy as np
from majority_vote import MajorityVoteClassifier, calc_error

# In House
from majority_vote import DataInterface

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
        val_percentage = len(np.extract(labels == lbl, labels)) / total_lbls
        entropy       -= val_percentage * log2(val_percentage)

    return entropy

def write_output(file, entropy, error):
    """Write the entropy and error to an output file

    Args:
        file (string): The file to write the data
        entropy (float): The entropy to write to the file
        error (float): The error value to write to the file
    """
    with open(file, "w+") as f:
        f.write(f"entropy: {entropy}\n")
        f.write(f"error: {error}\n")

def run_inspection(data, out_file):
    """Run an inspection on the data. Calculating error for MV Classifier and find the entropy

    Args:
        data (np.ndarray): The full array of features and labels
        out_file (string): The output file path for the inspection
    """
    lbls    = data[:, -1]
    entropy = calc_entropy(lbls)

    # Create a majority vote classifier and receive majority vote predictions
    mvc = MajorityVoteClassifier()
    mvc.fit(data)
    preds = mvc.infer()

    # Calculate the percent error given predictions and labels
    error = calc_error(preds, lbls)

    # Write the resulting entropy and error to the given file
    write_output(out_file, entropy, error)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        input  = sys.argv[1]
        output = sys.argv[2]

    # Get the data from the tsv file
    data_interface = DataInterface()
    data_interface.read_tsv(input)
    data = data_interface.get_data()

    # Run the inspection
    run_inspection(data, output)