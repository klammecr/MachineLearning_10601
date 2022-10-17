import csv
import numpy as np
import sys

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################

def process_data(in_file, out_file, in_feat):
    """Process the data into a feature embedding

    Args:
        in_file (str): The input dataset
        out_file (str): The output file for the embedding
        in_feat (dict): The file with the embeddings
    """
    dataset          = load_tsv_dataset(in_file)
    feature_mapping  = load_feature_dictionary(in_feat)
    embedded_dataset = embed_dataset(dataset, feature_mapping)
    write_output(embedded_dataset, out_file)

def embed_dataset(data, embeddings):
    """Take in a dataset and give a feature representation

    Args:
        data (np.ndarray): Vector of shape n data points
        embeddings (dict): A dictionary containing the word embeddings
    """
    # Output that is the number of samples x number of features
    feats = np.zeros((len(data), 301))

    for sample_idx, sample in enumerate(data):
        label, review = sample

        for word_num, word in enumerate(review.split(" ")):
            feat = embeddings.get(word)
            # Only add valid embeddings
            if feat is not None:
                feats[sample_idx, 1:] += feat

        # Average the response over all the words
        feats[sample_idx] /= (word_num + 1)

        # Add the label
        feats[sample_idx, 0] = label

    return feats

def write_output(data, out_file):
    """Format the output and write it to a file

    Args:
        data (np.ndarray): The data to write to file
        out_file (str): The file to write to
    """
    with open(out_file, "w+") as f:
        for row_num, row_data in enumerate(data):
            if row_num != 0:
                f.write("\n")
            for data_entry in row_data:
                f.write(f"{data_entry:.6f}")
                f.write("\t")


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc == 8:
        train_input     = sys.argv[1]
        val_input       = sys.argv[2]
        test_input      = sys.argv[3]
        feat_dict_input = sys.argv[4]
        train_out       = sys.argv[5]
        val_out         = sys.argv[6]
        test_out        = sys.argv[7]
        process_data(train_input, train_out, feat_dict_input)
        process_data(val_input, val_out, feat_dict_input)
        process_data(test_input, test_out, feat_dict_input)

    else:
        raise ValueError("Wrong Number of args")