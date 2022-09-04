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
            

            



if __name__ == "__main__":
    # Handle CLI arguments
    if len(sys.argv) == 6:
        train_input = sys.argv[1]
        test_input  = sys.argv[2]
        train_out   = sys.argv[3]
        test_out    = sys.argv[4]
        met_out     = sys.argv[5]
    else:
        raise ValueError("Please pass in five command line arguments: python majority_vote.py <train_input> <test_input> <train_out> <test_out> <metrics_out>")

    interface = DataInterface()
    interface.read_tsv("hw1/handout/heart_train.tsv")
    interface.get_data(["sex", "chest_pain"])