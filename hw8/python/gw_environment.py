import numpy as np

np.random.seed(0)

class Grid:
    """
    This class handles deterministic moving and the grid itself, but 
    `Gridworld` handles reward values, probabilistic moves, etc.
    """
    def __init__(self, r: int, c: int):
        """
        Initializes the grid at position `r`, `c` (rows, columns)
        """
        # Row and column positions
        self.r = r
        self.c = c

        # Height and width
        self.h = 3
        self.w = 4

        self.grid = np.array([["U", "V", "W", "X"],
                              ["P", "Q", "R", "S"],
                              ["K", "L", "M", "N"]])
        
        self.terminal_states = ("P", "X")
        self.blocked_states  = ("K", "R", "U")
    
    def move(self, direction: str):
        """
        Deterministically moves the cursor in the specified direction (one of 
        "up", "down", "left", "right"), or does nothing if at edge, or does 
        nothing if moving into a blocked state
        """
        # Store original in case new state is blocked
        r = self.r
        c = self.c

        # Define minimum and maximum rows/cols
        min_row = 0
        max_row = self.h - 1
        min_col = 0
        max_col = self.w - 1

        if   direction == "up":
            self.r = max(min_row, self.r - 1)
        elif direction == "down":
            self.r = min(max_row, self.r + 1)
        elif direction == "left":
            self.c = max(min_col, self.c - 1)
        elif direction == "right":
            self.c = min(max_col, self.c + 1)
        else:
            raise Exception(f"direction must be 'up', 'down', 'left', or 'right', got {direction}")
        
        if self.is_blocked(): # restore the original
            self.r = r
            self.c = c
    
    def moveto(self, r: int, c: int):
        """
        Moves the cursor to the specified row `r` and column `c`
        """
        self.r = r
        self.c = c
    
    def label(self) -> str:
        """
        Returns the label of the current state (e.g. "S", "K", "R", ...)
        """
        return self.grid[self.r, self.c]
    
    def show_class_mascot(self) -> str:
        """
        Returns the name of the class mascot
        """
        print("Neural the Narwhal!")
    
    def is_terminal(self) -> bool:
        """
        Checks to see if the cursor (`self.r`, `self.c`) is in a terminal state
        """
        return True if self.label() in self.terminal_states else False
    
    def is_blocked(self) -> bool:
        """
        Checks to see if the cursor (`self.r`, `self.c`) is in a blocked state
        """
        return True if self.label() in self.blocked_states else False
    
    def index(self) -> int:
        """
        Returns the index of the current state; each state has a unique index. 
        This is used later as the exposed state representation in `Gridworld`
        """
        return self.r * self.w + self.c

    


class Gridworld:
    """
    Gridworld class. Exposed interface:

        `__init__` : Initializes the gridworld
        `reset`    : Resets the gridworld to initial conditions
        `step`     : Take a step in the environment; set `done=True` when done
    """
    def __init__(self, mode: str, fixed=None):
        """
        Initializes the gridworld. For now, always initializes at position `M`. 
        Make sure to call `reset` immediately after initializing the gridworld. 
        The `fixed` argument is ignored (but it's there for consistency with 
        MountainCar), and `mode` must always be "tile"
        """
        if mode != "tile":
            raise Exception(f"You *must* use tile mode for Gridworld, not {mode}")
        
        self.grid = Grid(r=2, c=2) # pos M (Earth)
        self.state_space  = 3*4
        self.action_space = 4

        self.all_actions = ("up", "down", "left", "right")
        self.act_to_idx  = {"up": 0, "down": 1, "left": 2, "right": 3}
        self.idx_to_act  = {0: "up", 1: "down", 2: "left", 3: "right"}
        
        self._rng = np.random.default_rng(seed=0)
    
    def reset(self) -> dict:
        """
        Resets the gridworld to initial conditions
        """
        self.done = False
        self.grid.moveto(r=2, c=2)

        return {self.grid.index(): 1}

    def step(self, action: str) -> tuple:
        """
        Takes the action `action` in ("up", "down", "left", "right"), with 
        probabilistic transitions. Returns the state, reward, and a flag 
        indicating whether an episode is over or not. Note that the state 
        representation follows the Mountain Car environment's sparse 
        tile format.
        """
        # Determine the relative left and right moves
        if   action == "up":
            rel_L = "left"
            rel_R = "right"
        elif action == "down":
            rel_L = "right"
            rel_R = "left"
        elif action == "left":
            rel_L = "down"
            rel_R = "up"
        elif action == "right":
            rel_L = "up"
            rel_R = "down"
        else:
            raise Exception(f"Expected action to be one of 'up', 'down', 'left', 'right', but got {action}")

        # Make the move
        p_intended = 0.80
        p_left     = 0.10
        p_right    = 0.10

        from_state = self.grid.label() # for reward calculation

        move = self._rng.choice(a=[action, rel_L, rel_R],
                                p=[p_intended, p_left, p_right])
        self.grid.move(direction=move)

        # Compute the reward
        if   from_state == "W" and self.grid.label() == "X":
            reward = 100
        elif from_state == "S" and self.grid.label() == "X":
            reward = 10
        elif from_state == "Q" and self.grid.label() == "P":
            reward = -30
        else:
            reward = 0

        # Return the state, reward, and done flag
        state     = {self.grid.index(): 1}
        self.done = self.grid.is_terminal()

        return state, reward, self.done