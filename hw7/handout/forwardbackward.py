import argparse
import numpy as np

def get_inputs(args = None):
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    if not args:
        parser = argparse.ArgumentParser()
        
        parser.add_argument("validation_data", type=str)
        parser.add_argument("index_to_word", type=str)
        parser.add_argument("index_to_tag", type=str)
        parser.add_argument("hmminit", type=str)
        parser.add_argument("hmmemit", type=str)
        parser.add_argument("hmmtrans", type=str)
        parser.add_argument("predicted_file", type=str)
        parser.add_argument("metric_file", type=str)

        args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix
def logsumexp(inp):
    log_exp_sum = None
    if len(inp.shape) == 1:
        c = np.max(inp)
        log_exp_sum = c + np.log(np.sum(np.exp(inp - c)))
    elif len(inp.shape) == 2:
        # Center
        c = np.max(inp, axis = 1)
        # For each row, first exponentiate then sum
        exp_sum = np.sum(np.exp(inp - c.reshape(-1, 1)), axis = 1)
        log_exp_sum = c + np.log(exp_sum)

    else:
        raise ValueError("Undefined shape")

    return log_exp_sum

def compute_pred_tags(log_alpha, log_beta):
    """
    Compute predicted tags for the HMM.

    This is equivalent to:
    = argmax_y (P(Y_t = y|x)) for time t
    = argmax_y P(x, Y_t = y) / P(x) 
    """
    log_p_y_given_x       = log_alpha + log_beta
    most_probable_tag_idx = np.argmax(log_p_y_given_x, axis = 1)

    return most_probable_tag_idx

def compute_log_prob(log_alpha):
    """
    Compute the log probability of the sequence.

    This is equivalent to:
    P(x) = sum_y P(x, y)
    """

    return logsumexp(log_alpha[-1])

def forwardbackward(seq, loginit, logtrans, logemit, word_to_indices):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix
    
    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    seq_len           = len(seq)
    num_hidden_states = len(loginit)

    # Initialize log_alpha and fill it in
    log_alpha = np.zeros((seq_len, num_hidden_states)) # alpha for each entry in the sequence, an entry for each hidden state
    x_idx = word_to_indices[seq[0][0]]
    log_alpha[0] = loginit + logemit[:, x_idx]
    for t in range(1, seq_len):
        x_idx = word_to_indices[seq[t][0]]
        # Weight each row (prev state) by the alpha
        # Transpose because we need to sum the row
        inter = logsumexp((log_alpha[t-1].reshape(-1, 1) + logtrans).T)
        log_alpha[t] = inter + logemit[:, x_idx]

    # Initialize log_beta and fill it in
    log_beta = np.zeros((seq_len, num_hidden_states))
    for t in range(seq_len - 2, -1, -1):
        x_idx = word_to_indices[seq[t+1][0]]
        log_beta[t] = logsumexp((log_beta[t+1] + logtrans + logemit[:, x_idx]))

    # Compute the log-probability of the sequence
    log_likelihood_seq = compute_log_prob(log_alpha)

    # Compute the predicted tags for the sequence
    pred_tags = compute_pred_tags(log_alpha, log_beta)

    # Return the predicted tags and the log-probability
    return pred_tags, log_likelihood_seq 
    
if __name__ == "__main__":
    # Get the input data
    inputs = get_inputs()
    val_data        = inputs[0]
    words_to_indices = inputs[1]
    tags_to_indices = inputs[2]
    hmminit         = inputs[3] 
    hmmemit         = inputs[4]
    hmmtrans        = inputs[5]
    out_pred        = inputs[6]
    out_met         = inputs[7]

    # Invert the dictionary
    indices_to_tags = dict((v, k) for k, v in tags_to_indices.items())

    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.
    log_probs = []
    all_pred_tags = []
    tag_match = []
    for val_sequence in val_data:
        pred_idxs, log_prob = forwardbackward(val_sequence, np.log(hmminit), np.log(hmmtrans), np.log(hmmemit), words_to_indices)
        pred_tags = []
        for i, pred_idx in enumerate(pred_idxs):
            # Find out what the tag actually is by mapping it
            pred_tag = indices_to_tags[pred_idx]
            pred_tags.append(pred_tag)
            tag_match.append(pred_tag == val_sequence[i][1])

        # Save to arrays
        log_probs.append(log_prob)
        all_pred_tags.append(pred_tags)

    # Compute Average log-likelihood
    avg_ll = np.mean(np.array(log_probs))

    # Compute Accuracy
    tag_match = np.array(tag_match)
    accuracy = np.sum(tag_match) / len(tag_match)

    # Save to metrics file
    with open(out_met, "w") as f:
        f.write(f"Average Log-Likelihood: {avg_ll}\n")
        f.write(f"Accuracy: {accuracy}\n")

    # Save predictions to predictions file
    with open(out_pred, "w") as f:
        for seq_idx, seq in enumerate(all_pred_tags):
            for word_idx, tag in enumerate(seq):
                f.write(f"{val_data[seq_idx][word_idx][0]}\t{tag}\n")
            f.write("\n")