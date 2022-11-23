import argparse
import numpy as np


def get_inputs(args = None):
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    if not args:
        parser = argparse.ArgumentParser()
        parser.add_argument("train_input", type=str)
        parser.add_argument("index_to_word", type=str)
        parser.add_argument("index_to_tag", type=str)
        parser.add_argument("hmmprior", type=str)
        parser.add_argument("hmmemit", type=str)
        parser.add_argument("hmmtrans", type=str)

        args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans

def learn(inputs):
    # Collect the input data
    train_data         = inputs[0]
    word_to_indices    = inputs[1]
    tags_to_indices    = inputs[2]
    out_init_prob      = inputs[3]
    out_emit_matx      = inputs[4]
    out_transition_mtx = inputs[5]


    # Initialize the initial, emission, and transition matrices
    num_hidden_states = len(tags_to_indices)
    num_obs           = len(word_to_indices)

    init_prob      = np.ones((num_hidden_states))
    emit_mtx       = np.ones((num_hidden_states, num_obs))
    transition_mtx = np.ones((num_hidden_states, num_hidden_states))

    # Increment the matrices
    for train_sample in train_data:
        for w in range(len(train_sample)):
            # Extract information for the word
            word     = train_sample[w]
            tag_idx  = tags_to_indices[word[1]]
            word_idx = word_to_indices[word[0]]

            # Increment counts
            if w == 0:
                init_prob[tag_idx] += 1
            else:
                prev_tag_idx = tags_to_indices[train_sample[w-1][1]]
                transition_mtx[prev_tag_idx, tag_idx] += 1

            emit_mtx[tag_idx, word_idx] += 1

    # Turn into probabilities by normalizing
    init_prob /= np.sum(init_prob) # normalize by the total number of initial states
    transition_mtx /= np.sum(transition_mtx, axis=1).reshape(num_hidden_states,1) # normalize by the # of times the given previous hidden state was present
    emit_mtx /= np.sum(emit_mtx, axis=1).reshape(num_hidden_states,1) # normalize by the # of times the hidden state was present

    # Save the numpy arrays to text files
    np.savetxt(out_init_prob, init_prob, delimiter=" ")
    np.savetxt(out_emit_matx, emit_mtx, delimiter=" ")
    np.savetxt(out_transition_mtx, transition_mtx, delimiter=" ")

if __name__ == "__main__":
    inputs = get_inputs()
    learn(inputs)