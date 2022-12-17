
# Third Party
import sys
from math import log2
import numpy as np
import re

class DataInterface:
    def __init__(self):
        self.data   = None
        self.labels = None

    def read_tsv(self, file) -> np.ndarray:
        """
        Read the TSV file

        Args:
            file (string): The file path for the TSV file
        """
        with open(file, "r") as f:
            data = []
            lines = f.readlines()
            for idx, line in enumerate(lines):
                # Filter out the last column, that one will be empty because of the newline character
                features = re.split("[\n\t]", line)[:-1]

                # The first line is just labels, keep them as strings
                if idx == 0:
                    self.headers = features
                else:
                    features = [int(feature) for feature in features]
                    data.append(features) # fine for now, may want to support floats later
            self.data = np.array(data)

    def get_headers(self):
        return np.array(self.headers)

    def get_data(self, keys = None):
        """Return the data

        Args:
            keys (list, optional): A list of column names to grab. Defaults to None.

        Returns:
            np.ndarray: The numpy array holding the data
        """
        # IF there is no data, we can't return anything
        if self.data is None:
            return None

        # No specified keys, grab all the data
        if keys is None:
            return_data = self.data
        else:
            # Find the matching columns
            columns = np.zeros(self.data.shape[1], dtype=bool)
            for key in keys:
                columns[np.where(np.array(self.headers) == key)] = 1

            # Create a mask by repeating the 1s and 0s down all the rows
            columns     = np.resize(columns,  (1, len(columns)))
            data_mask   = np.repeat(columns, np.shape(self.data)[0], axis = 0)
            return_data = self.data[np.where(data_mask == 1)].reshape(self.data.shape[0], np.sum(columns[0, :]))

        return return_data

class MajorityVoteClassifier:
    """Create a classifier based on which label occurs most often
    """
    def __init__(self) -> None:
        self.cls = None

    def fit(self, lbls):
        """Train the classifier based on the data

        Args:
            data (np.ndarray): The data structure
        """
        self.cls = int(np.sum(lbls == 1) >= np.sum(lbls == 0))

    def infer(self):
        """Predict a classificaiton for the given data

        Args:
            data_entry (np.ndarray): The data (features) to base the prediciton on

        Returns:
            _type_: _description_
        """
        return self.cls

def write_labels(data, classifier, out_file):
    """Write the predicted labels for each entry in the dataset

    Args:
        data (np.ndarray): The data to write the predictions for
        classifier (MajorityVoteClassifier): The classifier to compute the predicted labels
        out_file (string): The file where to write the labels (delimited by \n)

    Raises:
        ValueError: _description_
    """
    with open(out_file, "w+") as f:
        for entry in data:
            pred = classifier.infer(entry)
            f.write(f"{pred}\n")

def read_preds(label_file):
    """Read the label prediction files

    Args:
        label_file (string): The label file to read

    Returns:
        np.ndarray: Labels in a numpy array
    """
    preds = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            preds.append(int(line[0]))
    return np.array(preds)

def calc_error_from_file(pred_file, labels):
    """Calculate the percent error for a given prediction and label set

    Args:
        pred_file (string): The file of label predictions
        labels (np.ndarray): The labels of the dataset

    Returns:    
        float: The error for the given set
    """
    preds = read_preds(pred_file)
    error = calc_error(preds, labels)
    return error

def calc_error(preds, labels):
    """Calculate the percent error given the predictions and labels

    Args:
        preds (np.ndarray): The predictions
        labels (np.ndarray): The labels (classes)

    Returns:
        float: percent error
    """
    return np.sum(np.abs(preds - labels)) / np.prod(np.size(labels))

def write_error(train_pred_file, train_labels, test_pred_file, test_labels, out_file):
    """Write the error of the test set and train set to a file

    Args:
        train_pred_file (string): The predicitons for the train set
        train_labels (np.ndarray): The labels for the train set
        test_pred_file (string): The predictions for the test set 
        test_labels (np.ndarray): The labels for the test set
        out_file (string): The outfile for the metrics
    """
    train_error = calc_error_from_file(train_pred_file, train_labels)
    test_error  = calc_error_from_file(test_pred_file, test_labels)
    with open(out_file, "w+") as f:
        f.write(f"error(train): {train_error}\n")
        f.write(f"error(test): {test_error}")

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