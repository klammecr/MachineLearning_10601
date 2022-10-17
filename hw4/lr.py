import numpy as np
import sys
from math import log, e

def process_inputs(train_in):
    """Process the inputs and get ready for training

    Args:
        train_in (str): File of the train set
    """
    # Load in the data
    train_set = np.loadtxt(train_in)

    # Calculate x with the bias foleded in
    X    = train_set[:, 1:]
    bias = np.ones((X.shape[0], 1))
    X    = np.hstack((bias, X))

    # Extract the labels
    y = train_set[:, 0]
    theta     = np.zeros((len(train_set[0])))
    return X, y, theta

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def plot_nll(train_nll, val_nll = None, lr = None):
    """Plot the negative log likelihood for each epoch

    Args:
        train_nll (np.ndarray): Negative log likelihood array of size num_epochs
        val_nll (np.ndarray): Negative log likelihood array of size num_epochs
    """
    import matplotlib.pyplot as plt
    plt.plot(list(range(len(train_nll))), train_nll, label=f"Learning Rate: {lr}")
    # plt.plot(list(range(len(val_nll))), val_nll, label = "Validation Set")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.title(f"Average NLL per Epoch")

def forward(theta, X, y, num_epoch, learning_rate, X_val = None, y_val = None):
    """Train the logistic regression model.

    Args:
        theta (np.ndarray): The array of weights 
        X (np.ndarray): The training set features
        y (np.ndarray): Labels for each sample
        num_epoch (int): Number of epochs to train for
        learning_rate (float): Leearning rate for updating the parameters via gradient ascent

    Returns:
        theta: The weights for the logistic regression model
    """
    avg_nll_train = []
    avg_nll_val   = []
    for epoch in range(num_epoch):
        print(f"Epoch: {epoch}")
        nll = []

        # Training
        for sample in range(X.shape[0]):
        
            # Calculate the dot product between the X vector and the weights
            # This should be the size of the number of samples, each sample is weighted and added up over all features
            weighted_x =  np.dot(X[sample], theta)
            # weighted_x = np.dot(theta, np.hstack((bias, X)))

            # Calculate the gradient for each feature, this should be the gradient for each feature
            g = X[sample] * (sigmoid(weighted_x) - y[sample])

            # Perform a gradient ascent step:
            theta = theta - learning_rate * g
            
            nll.append(-1 * (y[sample] * weighted_x - log(1 + e**(weighted_x))))

        # Add to the avgerage negative log likelihood tracking
        avg_nll_train.append(np.array(nll).mean())
        nll = []

        # Validation
        if X_val is not None and y_val is not None:
            for sample in range(X_val.shape[0]):
                weighted_x =  np.dot(X_val[sample], theta)
                nll.append(-1 * (y_val[sample] * weighted_x - log(1 + e**(weighted_x))))

         # Add to the avgerage negative log likelihood tracking
        avg_nll_val.append(np.array(nll).mean())
        nll = []

    return theta, np.array(avg_nll_train), np.array(avg_nll_val)

def predict(theta, X):
    pred_probs = X @ np.expand_dims(theta, axis=1)
    preds      = sigmoid(pred_probs) >= 0.5
    return preds.astype("float")

def compute_error(y_pred, y):
    """Compute the error for the given set

    Args:
        y_pred (np.ndarray): Vector of predictions
        y (np.ndarray): Vector of truth labels
    """
    return np.sum((y_pred.reshape(-1) != y.reshape(-1))) / len(y)

def write_error(file, error_train, error_test):
    """Write the error value to the given file

    Args:
        file (string): The file path
        error_train (float): Training error
        error_test (float): Test error
    """
    with open(file, "w+") as f:
        f.write(f"error(train): {error_train:.6f}\n")
        f.write(f"error(test): {error_test:.6f}")

def write_labels(file, data):
    """Write the labels to a specified file for grading

    Args:
        file (string): The file to write the labels to 
        data (np.ndarray): The labels to write
    """
    with open(file, "w+") as f:
        for label in data:
            f.write(str(int(label)) + "\n")

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc == 9:
        train_input     = sys.argv[1]
        val_input       = sys.argv[2]
        test_input      = sys.argv[3]
        train_out       = sys.argv[4]
        test_out        = sys.argv[5]
        met_out         = sys.argv[6]
        num_epoch       = int(sys.argv[7])
        lr              = float(sys.argv[8])
        
        # lrs = [10e-3, 10e-4, 10e-5]
        lrs = [lr]
        for lr in lrs:
            # Process the inputs into X, y, and theta
            X, y, theta     = process_inputs(train_input)
            X_val, y_val, _ = process_inputs(val_input)

            # Train the logistic regression model
            X = np.array([[0,0,1], [0,1,0], [0,1,1], [1,0,0]])
            theta = np.array([1.5, 2, 1])
            y = np.array([0,1,1,0])
            theta, avg_nll_train, avg_nll_val = forward(theta, X, y, num_epoch, lr, X_val, y_val)

            # Question 7: Programming Emprirical Questions
            # plot_nll(avg_nll_train, avg_nll_val)
            # plot_nll(avg_nll_train, lr=lr)

            # Run Predictions on train set
            preds_train = predict(theta, X)
            write_labels(train_out, preds_train)
            train_error = compute_error(preds_train, y)

            # Do prediction on the test set
            X_test, y_test, _ = process_inputs(test_input)
            preds = predict(theta, X_test)
            write_labels(test_out, preds)
            test_error = compute_error(preds, y_test)

            # Write the metrics to a file
            write_error(met_out, train_error, test_error)



    else:
        raise ValueError("Wrong Number of args")    
