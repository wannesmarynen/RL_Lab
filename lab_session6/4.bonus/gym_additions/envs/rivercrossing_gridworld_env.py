import gym
from gym import spaces
import numpy as np


class RiverCrossingEnv(gym.Env):
    """ Small 3x5 Gridworld with a river in the middle row."""

    def __init__(self):
        self.height = 3
        self.width = 5
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.height),
            spaces.Discrete(self.width)
        ))
        self.moves = {
            0: (-1, 0),  # up
            1: (0, 1),  # right
            2: (1, 0),  # down
            3: (0, -1),  # left
            4: (0, 0),
        }

        self.grid = [["", "", "", "", "E"],
                     ["w", "w", "w", "D"],
                     ["S", "", "", "", ""]]

        self.state = (0, 0)

    def reset(self):
        """
        Resets the environment to the first state of the environment.
        Returns
        -------
        state : State
            First state of the episode.
        """
        # TODO: Implement this
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.height),
            spaces.Discrete(self.width)
        ))
        self.state = (0, 0)
        return self.state

    def step(self, action):
        """
        Performs a single step of the environment given an action.
        i.e. samples s',r from the p(s',r | s,a) distribution.
        Parameters
        ----------
        action: int
            The action performed by the agent
        Returns
        -------
        next_state : state
            State resulting from the transition
        reward : float
            Reward for this transition
        done : bool
            Whether next_state is a terminal state.
        info : dict
            ignore this
        """
        # TODO: code here
        x, y = self.state
        if action == 4 and self.grid[x][y] != 'w':
            return self.state, 0, False, ""
        else:
            new_state = (self.state[0] + self.moves[action][0], self.state[1] + self.moves[action][1])
            if new_state[0] < 0 or new_state[0] >= self.height or new_state[1] >= self.width or new_state[1] < 0:
                return self.state, 0, False, ""
            else:
                x, y = new_state
                if self.grid[x][y] == 'w':
                    return new_state, 0, False, ""
                elif self.grid[x][y] == 'E':
                    return new_state, 1, True, ""
                else:
                    return new_state, 0, False, ""


    def render(self):
        """
        Optional.
        Prints the environment for visualization.
        """
        pass
