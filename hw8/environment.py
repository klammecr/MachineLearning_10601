# STUDENTS: DO NOT CHANGE THIS FILE! GRADESCOPE WILL REVERT ANY CHANGES YOU MAKE
# HERE!

import math
import logging
import numpy as np
from tiles import tiles, IHT


# Environment
class MountainCar:
    def __init__(self, mode=None, debug=False):
        # Initial positions of box-car
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        
        self.force = 0.001
        self.gravity = 0.0025

        # Actions = {0, 1, 2} for go left, do nothing, go right
        self.action_space = 3

        if mode == "tile":
            self.state_space = 2048
        elif mode == 'raw':
            self.state_space = 2
        else:
            raise Exception("Invalid environment mode. Must be tile or raw")

        self.mode = mode
        # variables used conditionally on mode or render
        self.iht = None
        self.w = None
        self.viewer = None  # needed for render only
        if debug:
            logging.basicConfig(
                format=('[%(asctime)s] {%(pathname)s:%(funcName)s:'
                        '%(lineno)04d} %(levelname)s - %(message)s'),
                datefmt='%H:%M:%S',
                level=logging.DEBUG)
        self.debug = debug
        self.reset()

    def transform(self, state):
        # Normalize values to range from [0, 1] for use in transformations
        position, velocity = state
        position = (position + 1.2) / 1.8
        velocity = (velocity + 0.07) / 0.14
        assert 0 <= position <= 1
        assert 0 <= velocity <= 1
        position *= 2
        velocity *= 2
        if self.mode == "tile":
            if self.iht is None:
                self.iht = IHT(self.state_space)
            tiling = tiles(self.iht, 64, [position, velocity], [0]) + \
                     tiles(self.iht, 64, [position], [1]) + \
                     tiles(self.iht, 64, [velocity], [2])
            return_state = np.zeros(self.state_space, dtype=float)
            for index in tiling:
                return_state[index] = 1
            return return_state
        elif self.mode == "raw":
            return state.copy()
        else:
            raise Exception("Invalid environment mode. Must be tile or raw")

    def reset(self):
        self.state = np.array([-0.5, 0])
        retval = self.transform(self.state)
        if self.debug:  # to avoid unwanted slowdown
            logging.debug(f'Reset: {retval}')
        return retval
    
    def height(self, xs):
        return np.sin(3 * xs)*.45+.55
    
    def step(self, action):
        assert action in {0, 1, 2}

        position, velocity = self.state
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        # Left of min_position is a wall
        if position == self.min_position and velocity < 0:
            velocity = 0
        
        done = position >= self.goal_position
        reward = -1.0

        self.state = np.array([position, velocity])
        state = self.transform(self.state)
        if self.debug:  # to avoid unwanted slowdown
            logging.debug(f'Step (action {action}): state {state}, '
                          f'reward {reward}, done {done}')
        return state, reward, done

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self.height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self.height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10),
                 (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos - self.min_position) * scale,
                                      self.height(pos) * scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


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

        self.grid = np.array([["I", "J", "K", "L"],
                              ["E", "F", "G", "H"],
                              ["A", "B", "C", "D"]])

        self.terminal_states = ("E", "L")
        self.blocked_states = ("A", "G", "I")

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

        if direction == "up":
            self.r = max(min_row, self.r - 1)
        elif direction == "down":
            self.r = min(max_row, self.r + 1)
        elif direction == "left":
            self.c = max(min_col, self.c - 1)
        elif direction == "right":
            self.c = min(max_col, self.c + 1)
        else:
            raise Exception(
                f"direction must be 'up', 'down', 'left', or 'right', got {direction}")

        if self.is_blocked():  # restore the original
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

    def loc(self, label: str) -> tuple:
        """
        Inverse of `label`: given a string label, returns the (r, c) location
        """
        for r in range(self.h):
            for c in range(self.w):
                if self.grid[r, c] == label: return r, c

        raise KeyError(f"label passed into loc is invalid: {label}")

    def show_class_mascot(self) -> None:
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


class GridWorld:
    """
    Gridworld class. Exposed interface:

        `__init__` : Initializes the gridworld
        `reset`    : Resets the gridworld to initial conditions
        `step`     : Take a step in the environment; set `done=True` when done
    """

    def __init__(self, mode: str, debug=False):
        """
        Initializes the gridworld. For now, always initializes at position `C`.
        Make sure to call `reset` immediately after initializing the gridworld.
        The `fixed` argument is ignored (but it's there for consistency with
        MountainCar), and `mode` must always be "tile".
        """
        if mode != "tile":
            raise Exception(
                f"You *must* use tile mode for Gridworld, not {mode}")

        self.grid = Grid(r=2, c=2)  # pos C (Earth)
        self.state_space = 3 * 4
        self.action_space = 4

        self.all_actions = ("up", "down", "left", "right")
        self.act_to_idx = {"up": 0, "down": 1, "left": 2, "right": 3}
        self.idx_to_act = {0: "up", 1: "down", 2: "left", 3: "right"}

        self._rng = np.random.default_rng(seed=0)
        if debug:
            logging.basicConfig(
                format=('[%(asctime)s] {%(pathname)s:%(funcName)s:'
                        '%(lineno)04d} %(levelname)s - %(message)s'),
                datefmt='%H:%M:%S',
                level=logging.DEBUG)
        self.debug = debug
        self.done = False
        self.reset()

    def reset(self) -> dict:
        """
        Resets the gridworld to initial conditions
        """
        self.done = False
        self.grid.moveto(r=2, c=2)
        retval = np.zeros(self.state_space, dtype=float)
        retval[self.grid.index()] = 1
        if self.debug:  # to avoid unwanted slowdown
            logging.debug(f'Reset: {retval}')
        return retval

    def step(self, action: int) -> tuple:
        """
        Takes the action `action` in ("up", "down", "left", "right"), with
        probabilistic transitions. Returns the state, reward, and a flag
        indicating whether an episode is over or not. Note that the state
        representation follows the Mountain Car environment's sparse
        tile format.
        """
        # Determine the relative left and right moves
        try:
            action = self.idx_to_act[action]
        except KeyError:
            raise Exception(
                f"Expected action to be one of 'up', 'down', 'left', 'right', but got {action}")

        if action == "up":
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

        # Make the move
        p_intended = 0.80
        p_left = 0.10
        p_right = 0.10

        from_state = self.grid.label()  # for reward calculation

        move = self._rng.choice(a=[action, rel_L, rel_R],
                                p=[p_intended, p_left, p_right])
        self.grid.move(direction=move)

        # Compute the reward
        if from_state == "K" and self.grid.label() == "L":
            reward = 100
        elif from_state == "H" and self.grid.label() == "L":
            reward = 50
        elif from_state == "F" and self.grid.label() == "E":
            reward = -100
        else:
            reward = 0

        # Return the state, reward, and done flag
        state = np.zeros(self.state_space, dtype=float)
        state[self.grid.index()] = 1
        self.done = self.grid.is_terminal()

        if self.debug:  # to avoid unwanted slowdown
            logging.debug(f'Step (action {action}): state {state}, '
                          f'reward {reward}, done {self.done}')
        return state, reward, self.done

    def render(self, *args, **kwargs):
        print('Render is only implemented for the Mountain Car environment')

    def p(self, state_new, state, action):
        """
        p(s' | s, a)
        """
        if action == "up":
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
            raise Exception(
                f"Expected action to be one of 'up', 'down', 'left', 'right', but got {action}")

        # Make the move
        p_intended = 0.80
        p_left = 0.10
        p_right = 0.10

        # Try making moves
        r_original, c_original = self.grid.r, self.grid.c
        self.grid.moveto(*self.grid.loc(state))

        r, c = self.grid.r, self.grid.c
        cur_label = self.grid.label()

        self.grid.move(action)
        intended_label = self.grid.label()
        self.grid.moveto(r, c)

        self.grid.move(rel_L)
        L_label = self.grid.label()
        self.grid.moveto(r, c)

        self.grid.move(rel_R)
        R_label = self.grid.label()
        self.grid.moveto(r, c)

        self.grid.moveto(r_original, c_original)

        p_cur = 0
        if intended_label == cur_label:
            p_cur += p_intended
        if L_label == cur_label:
            p_cur += p_left
        if R_label == cur_label:
            p_cur += p_right

        p_i = p_intended if intended_label != cur_label else 0
        p_l = p_left if L_label != cur_label else 0
        p_r = p_right if R_label != cur_label else 0

        if state_new == cur_label:
            return p_cur
        elif state_new == L_label:
            return p_l
        elif state_new == R_label:
            return p_r
        elif state_new == intended_label:
            return p_i
        else:
            return 0

    def R(self, state, action, state_new):
        if state == "K" and state_new == "L":
            return 100
        elif state == "H" and state_new == "L":
            return 50
        elif state == "F" and state_new == "E":
            return -100
        else:
            return 0