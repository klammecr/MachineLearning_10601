import sys
from mc_environment import MountainCar
from gw_environment import GridWorld
import numpy as np
from random import random, randint

def main(args):
    # Extract the arguments we need
    env            = args[1]
    mode           = args[2]
    weight_out     = args[3]
    returns_out    = args[4]
    episodes       = args[5]
    max_iterations = args[6]
    epsilon        = args[7]
    gamma          = args[8]
    lr             = args[9]

    # Decide the environment
    if env == "mc":
        env = MountainCar(mode)
    elif env == "gw":
        env = GridWorld()
    else:
        raise ValueError("Invalid Env")

def get_qa_vals(weights, state):
    qa_w_bias = weights.T @ np.concatenate((np.array([1]), state))
    qa_vals   = qa_w_bias[1:] + qa_w_bias[0]
    return qa_vals

def get_best_action(weights, state):
    state_concat = np.concatenate((np.array([1]), state))
    acts         = weights.T @ state_concat
    act          = np.argmax(acts[1:])
    return act

def run_q_learning(
    env      : MountainCar or GridWorld,\
    episodes : int,
    max_iter : int,
    epsilon  : float,
    gamma    : float,
    lr       : float):

    # Get some initial information
    num_actions = env.all_actions.count
    num_states  = len(env.state_space)

    # Initialize the parameters, one for each action plus bias
    weights = np.zeros((num_states, num_actions + 1))
    
    # Loop for q learning
    for episode in range(episodes):
        for iter in range(max_iter):
            # Epsilon greedy: with probability epsilon take a random action
            if random() <= epsilon:
                # Sample a random action
                act = randint(0, env.all_actions.count - 1)
            else:
                # Take a greedy action according to the best q value
                act = get_best_action(weights, env.state)
            
            # Get the current q value
            state = env.state
            q_val = get_qa_vals(weights, state)[act]

            # Take the action
            next_state, reward, done = env.step(act)

            # Find the best q value for the next state
            qa_vals_next    = get_qa_vals(weights, next_state)
            best_q_val_next = np.max(qa_vals_next)

            # Calculate the target for TD
            target = reward + gamma * best_q_val_next

            # Calculate the TD error
            td_error = q_val - target

            # Perform the weight update
            weights[act] += - lr * td_error * state



if __name__ == "__main__":
    main(sys.argv)
