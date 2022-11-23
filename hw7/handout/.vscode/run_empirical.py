# Imports
from forwardbackward import get_inputs as fb_get_inputs
from forwardbackward import forwardbackward
from learnhmm import get_inputs as learnhmm_get_inputs
from learnhmm import learn
import numpy as np
import argparse
from copy import deepcopy

# Arg Parser
parser = argparse.ArgumentParser()

parser.add_argument("train_input", type=str)
parser.add_argument("index_to_word", type=str)
parser.add_argument("index_to_tag", type=str)
parser.add_argument("hmminit", type=str)
parser.add_argument("hmmemit", type=str)
parser.add_argument("hmmtrans", type=str)
parser.add_argument("validation_data", type=str)
parser.add_argument("predicted_file", type=str)
parser.add_argument("metric_file", type=str)



args = parser.parse_args()
args.hmmprior = args.hmminit

# Get the inputs
fb_inp = fb_get_inputs(args)
lh_inp = learnhmm_get_inputs(args)

# Set up arrays to be plotted
seq = [10, 100, 1000, 10000]
log_seq = np.log(np.array(seq))
val_accuracies = []
val_avg_lls = []
train_accuracies = []
train_avg_lls = []

for num_seq in seq:
    # Copy the training inputs then change the number of training samples
    lh_inp_copy = list(deepcopy(lh_inp))
    lh_inp_copy[0] = lh_inp_copy[0][:num_seq]

    # Learn the parameters
    learn(lh_inp_copy)

    # Reload the files every time
    fb_inp = fb_get_inputs(args)

    # Run the forward backward algo for inference
    train_data      = lh_inp[0]
    val_data        = fb_inp[0]
    words_to_indices= fb_inp[1]
    tags_to_indices = fb_inp[2]
    hmminit         = fb_inp[3] 
    hmmemit         = fb_inp[4]
    hmmtrans        = fb_inp[5]
    out_pred        = fb_inp[6]
    out_met         = fb_inp[7]

    # Invert the dictionary
    indices_to_tags = dict((v, k) for k, v in tags_to_indices.items())

    # Compute training stuff
    train_log_probs = []
    train_all_pred_tags = []
    tag_match = []
    for train_sequence in train_data:
        pred_idxs, log_prob = forwardbackward(train_sequence, np.log(hmminit), np.log(hmmtrans), np.log(hmmemit), words_to_indices)
        pred_tags = []
        for i, pred_idx in enumerate(pred_idxs):
            # Find out what the tag actually is by mapping it
            pred_tag = indices_to_tags[pred_idx]
            pred_tags.append(pred_tag)
            tag_match.append(pred_tag == train_sequence[i][1])

        # Save to arrays
        train_log_probs.append(log_prob)
        train_all_pred_tags.append(pred_tags)

    # Compute Average log-likelihood
    train_avg_ll = np.mean(np.array(train_log_probs))

    # Compute Accuracy
    tag_match    = np.array(tag_match)
    train_accuracy = np.sum(tag_match) / len(tag_match)

    # Add sequence results to array
    train_accuracies.append(train_accuracy)
    train_avg_lls.append(train_avg_ll)

    ####################################
    # Validation
    ####################################
    # Compute validation stuff
    val_log_probs = []
    val_all_pred_tags = []
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
        val_log_probs.append(log_prob)
        val_all_pred_tags.append(pred_tags)

    # Compute Average log-likelihood
    val_avg_ll = np.mean(np.array(val_log_probs))

    # Compute Accuracy
    tag_match    = np.array(tag_match)
    val_accuracy = np.sum(tag_match) / len(tag_match)

    # Add sequence results to array 
    val_accuracies.append(val_accuracy)
    val_avg_lls.append(val_avg_ll)
    pass

# Print the arrays for the table
print(f"Training Accuracies: {train_accuracies}")
print(f"Training Average LL: {train_avg_lls}")
print(f"Validation Accuracies: {val_accuracies}")
print(f"Validation Average LL: {val_avg_lls}")

# Create the plot
import matplotlib.pyplot as plt
plt.plot(log_seq, train_avg_lls, label = "Average LL for Training")
plt.plot(log_seq, val_avg_lls, label = "Average LL for Validation")
plt.xlabel("Number of Word Sequences")
plt.ylabel("Average Log-Likelihood")
plt.title("Effect of Number of Training Word Sequence on Average Log-Likelihood")
plt.legend()
pass