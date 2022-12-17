import argparse
import numpy as np
from random import randint, random

from environment import MountainCar, GridWorld

"""
Please read: THE ENVIRONMENT INTERFACE

In this homework, we provide the environment (either MountainCar or GridWorld) 
to you. The environment returns states, represented as 1D numpy arrays, rewards, 
and a Boolean flag indicating whether the episode has terminated. The environment 
accepts actions, represented as integers.

The only file you need to modify/read is this one. We describe the environment 
interface below.

class Environment: # either MountainCar or GridWorld

    def __init__(self, mode, debug=False):
        Initialize the environment with the mode, which can be either "raw" 
        (for the raw state representation) or "tile" (for the tiled state 
        representation). The raw state representation contains the position and 
        velocity; the tile representation contains zeroes for the non-active 
        tile indices and ones for the active indices. GridWorld must be used in 
        tile mode. The debug flag will log additional information for you; 
        make sure that this is turned off when you submit to the autograder.

        self.state_space = an integer representing the size of the state vector
        self.action_space = an integer representing the range for the valid actions

        You should make use of env.state_space and env.action_space when creating 
        your weight matrix.

    def reset(self):
        Resets the environment to initial conditions. Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the initial state.
    
    def step(self, action):
        Updates itself based on the action taken. The action parameter is an 
        integer in the range [0, 1, ..., self.action_space). Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the new state that the agent is in after taking its 
                        specified action.
            
            (2) reward : A float indicating the reward received at this step.

            (3) done : A Boolean flag indicating whether the episode has 
                        terminated; if this is True, you should reset the 
                        environment and move on to the next episode.
    
    def render(self, mode="human"):
        Renders the environment at the current step. Only supported for MountainCar.


For example, for the GridWorld environment, you could do:

    env = GridWorld(mode="tile")

Then, you can initialize your weight matrix to all zeroes with shape 
(env.action_space, env.state_space+1) (if you choose to fold the bias term in). 
Note that the states returned by the environment do *not* have the bias term 
folded in.
"""

def parse_args() -> tuple:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate
    
    Usage:
        env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)

    args = parser.parse_args()

    return args.env, args.mode, args.weight_out, args.returns_out, args.episodes, args.max_iterations, args.epsilon, args.gamma, args.learning_rate


def get_qa_vals(weights, state):
    state_concat = np.concatenate((np.array([1.0], dtype=np.float64), state))
    qa_vals = weights @ state_concat
    #qa_vals = np.dot(weights, state_concat)
    return qa_vals

def get_best_action(weights, state):
    qa_vals = get_qa_vals(weights, state)
    act     = np.argmax(qa_vals)
    return act

def run_q_learning(
    env      : MountainCar or GridWorld,\
    episodes : int,
    max_iter : int,
    epsilon  : np.float64,
    gamma    : np.float64,
    lr       : np.float64,
    out_wgt  : str,
    out_rwd  : str,
    plot_file: str = None):

    # Get some initial information
    num_actions = env.action_space
    num_states  = env.state_space

    # Initialize the parameters, one for each action plus bias
    weights = np.zeros((num_actions, 1 + num_states), dtype=np.float64)

    # Loop for q learning
    returns = []
    for episode in range(episodes):
        # Track the return for the episode
        sum_reward = 0.0

        # Reset to the beginning on the environment
        state = env.reset()

        for iter in range(max_iter):

            # Epsilon greedy: with probability epsilon take a random action
            if random() <= epsilon:
                # Sample a random action
                act = randint(0, num_actions - 1)
            else:
                # Take a greedy action according to the best q value
                act = get_best_action(weights, state)

            # Take the action
            next_state, reward, done = env.step(act)
            sum_reward += reward

            # Find the best q value for the next state
            qa_vals_next    = get_qa_vals(weights, next_state)
            best_next_act   = get_best_action(weights, next_state)
            best_q_val_next = qa_vals_next[best_next_act]

            # Calculate the target for TD
            target = reward + gamma * best_q_val_next

            # Calculate the TD error
            td_error = get_qa_vals(weights, state)[act] - target

            # Perform the weight update
            concat_state = np.concatenate((np.array([1.0], dtype=np.float64), state))
            weights[act] = weights[act] - lr * td_error * concat_state

            # The next state is now the current state
            state = next_state

            # Break if done
            if done:
                break
        
        # Append the return
        returns.append(sum_reward)
        
    # Save to text file
    np.savetxt(out_rwd, np.array(returns), fmt="%.18e", delimiter=" ")
    # Transpose is, we want each action to be a row
    np.savetxt(out_wgt, weights,    fmt="%.18e", delimiter=" ")

    # Plot the Results
    if plot_file is not None:
        from matplotlib import pyplot as plt
        from math import ceil
        t = list(range(0, len(returns)))

        num_ret = len(returns)
        rolling_avg_ret = np.zeros_like(returns)
        for window in range(ceil(num_ret / 25)):
            start_idx = window*25
            end_idx   = min((window+1)*25, num_ret)
            window_returns = returns[start_idx:end_idx]
            rolling_avg_ret[start_idx:end_idx] = np.cumsum(window_returns) / np.linspace(1, len(window_returns), len(window_returns))
        

        plt.plot(t, returns, label = "Episodic Returns")
        plt.plot(t, rolling_avg_ret, label="Rolling Mean Returns")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("MC Environment - Effect of Episode on Returns")
        plt.legend()
        plt.savefig(plot_file)
        plt.close()

if __name__ == "__main__":
    # Parse the args
    env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()

    # Decide the environment
    if env_type == "mc":
        env = MountainCar(mode)
    elif env_type == "gw":
        env = GridWorld(mode)
    else:
        raise ValueError("Invalid Env")

    #run_q_learning(env, episodes, max_iterations, epsilon, gamma, lr, weight_out, returns_out, "yo.png")

    # Plot of Raw
    run_q_learning(MountainCar("raw"), 2000, 200, 0.05, 0.999, 0.001, "1.txt", "2.txt", "empirical_raw.png")

    # Plot of Tile
    run_q_learning(MountainCar("tile"), 400, 200, 0.05, 0.99, 0.00005, "3.txt", "4.txt", "empirical_tile.png")
