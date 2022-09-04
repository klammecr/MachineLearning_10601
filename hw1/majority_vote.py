import re
import numpy as np
import sys

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
                    self.labels = features
                else:
                    features = [int(feature) for feature in features]
                    data.append(features) # fine for now, may want to support floats later
            self.data = np.array(data)

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
                columns[np.where(np.array(self.labels) == key)] = 1

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

    def fit(self, data):
        """Train the classifier based on the data

        Args:
            data (np.ndarray): The data structure
        """
        labels   = data[:, -1]
        self.cls = int(np.sum(labels == 1) >= np.sum(labels == 0))

    def infer(self, data_entry):
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

def calc_error(pred_file, labels):
    """Calculate the percent error for a given prediction and label set

    Args:
        pred_file (string): The file of label predictions
        labels (np.ndarray): The labels of the dataset

    Returns:    
        float: The error for the given set
    """
    preds = read_preds(pred_file)
    error = np.sum(np.abs(preds - labels)) / np.prod(np.size(labels))
    return error

def write_error(train_pred_file, train_labels, test_pred_file, test_labels, out_file):
    """Write the error of the test set and train set to a file

    Args:
        train_pred_file (string): The predicitons for the train set
        train_labels (np.ndarray): The labels for the train set
        test_pred_file (string): The predictions for the test set 
        test_labels (np.ndarray): The labels for the test set
        out_file (string): The outfile for the metrics
    """
    train_error = calc_error(train_pred_file, train_labels)
    test_error  = calc_error(test_pred_file, test_labels)
    with open(out_file, "w+") as f:
        f.write(f"error(train): {train_error}\n")
        f.write(f"error(test): {test_error}")


if __name__ == "__main__":
    # 1.2.3: Handle CLI arguments
    if len(sys.argv) == 6:
        train_input = sys.argv[1]
        test_input  = sys.argv[2]
        train_out   = sys.argv[3]
        test_out    = sys.argv[4]
        met_out     = sys.argv[5]
    else:
        raise ValueError("Please pass in five command line arguments: python majority_vote.py <train_input> <test_input> <train_out> <test_out> <metrics_out>")

    # 1.2.2: Read the data and format it to be ussable by the algorithm
    interface = DataInterface()
    interface.read_tsv(train_input)
    train_data = interface.get_data()

    # Train the classifier
    mv_cls = MajorityVoteClassifier()
    mv_cls.fit(train_data)

    # Grab the test set
    interface.read_tsv(test_input)
    test_data = interface.get_data()

    # 1.2.4: Write the predicted labels to a file for the training set and the test set
    write_labels(train_data, mv_cls, train_out)
    write_labels(test_data, mv_cls, test_out)

    # 1.2.5: Calculate the metrics and write them to a file
    write_error(train_out, train_data[:, -1], test_out, test_data[:, -1], met_out)