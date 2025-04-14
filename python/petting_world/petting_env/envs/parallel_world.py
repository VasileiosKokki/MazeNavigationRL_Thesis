import functools
import random
from copy import copy
from enum import Enum

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from gymnasium.utils import seeding

from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector

class Actions(Enum):
    right = 1
    up = 2
    left = 0
    down = 3
    still = 4


class CustomActionMaskedEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "custom_environment_v0",
        "render_fps": 3
    }

    possible_agents = ["prisoner", "guard"]

    def __init__(self, render_mode=None):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - escape x and y coordinates
        - guard x and y coordinates
        - prisoner x and y coordinates
        - timestamp
        - possible_agents

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.size = 7
        self.window_size = 512

        self.escape_y = None
        self.escape_x = None
        self.guard_y = None
        self.guard_x = None
        self.prisoner_y = None
        self.prisoner_x = None
        self.timestep = None
        self.possible_agents = ["prisoner", "guard"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        # self._agent_selector = agent_selector(self.possible_agents)

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, -1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, 1]),
            Actions.still.value: np.array([0, 0]),
        }

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.agents = copy(self.possible_agents)
        # self._agent_selector.reinit(self.agents)
        # self.agent_selection = self._agent_selector.next()
        self.timestep = 0


        self.prisoner_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        while True:
            self.guard_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if not np.array_equal(self.guard_location, self.prisoner_location):
                break

        while True:
            self.escape_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if not np.array_equal(self.escape_location, self.prisoner_location) and not np.array_equal(self.escape_location, self.guard_location):
                break

        observations = self.get_obs()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        # Execute actions
        # agent = self.agent_selection
        # is_last = self._agent_selector.is_last()
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # print(actions)
        if "prisoner" in actions and "guard" in actions:
            prisoner_action = actions["prisoner"]
            guard_action = actions["guard"]
        else:
            prisoner_action = -1
            guard_action = -1


        # print(actions)
        # print(self.prisoner_x)

        # if prisoner_action == 0 and self.prisoner_x > 0:
        #     self.prisoner_x -= 1
        # elif prisoner_action == 1 and self.prisoner_x < 6:
        #     self.prisoner_x += 1
        # elif prisoner_action == 2 and self.prisoner_y > 0:
        #     self.prisoner_y -= 1
        # elif prisoner_action == 3 and self.prisoner_y < 6:
        #     self.prisoner_y += 1
        #
        # if guard_action == 0 and self.guard_x > 0:
        #     self.guard_x -= 1
        # elif guard_action == 1 and self.guard_x < 6:
        #     self.guard_x += 1
        # elif guard_action == 2 and self.guard_y > 0:
        #     self.guard_y -= 1
        # elif guard_action == 3 and self.guard_y < 6:
        #     self.guard_y += 1

        prisoner_direction = self._action_to_direction[prisoner_action]
        # print(prisoner_direction,"prisoner")
        new_prisoner_location = np.clip(
            self.prisoner_location + prisoner_direction, 0, self.size - 1
        )
        guard_direction = self._action_to_direction[guard_action]
        # print(guard_direction,"guard")
        new_guard_location = np.clip(
            self.guard_location + guard_direction, 0, self.size - 1
        )

        prisoner_reward = 0
        guard_reward = 0
        if np.array_equal(self.prisoner_location, new_prisoner_location):
            prisoner_reward += -0.05

        if np.array_equal(self.guard_location, new_guard_location):
            guard_reward += -0.05

        self.prisoner_location = new_prisoner_location
        self.guard_location = new_guard_location
        # print(self.prisoner_location)
        # print(self.guard_location)
        # self.prisoner_location = np.array([self.prisoner_x, self.prisoner_y])
        # self.guard_location = np.array([self.guard_x, self.guard_y])
        # self.escape_location = np.array([self.escape_x, self.escape_y])
        # Check termination conditions
        terminations = {a: False for a in self.agents}
        if np.array_equal(self.prisoner_location, self.guard_location):
            prisoner_reward = -1
            guard_reward = 1
            terminations = {a: True for a in self.agents}
        elif np.array_equal(self.prisoner_location, self.escape_location):
            prisoner_reward = 1
            guard_reward = -1
            terminations = {a: True for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            prisoner_reward = 0
            guard_reward = 0
            truncations = {"prisoner": True, "guard": True}
        self.timestep += 1

        rewards = {"prisoner": prisoner_reward, "guard": guard_reward}
        # Get observations
        observations = self.get_obs()

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        # print("Terminations shape:", terminations)
        # print("Truncations shape:", truncations)

        # self.agent_selection = self._agent_selector.next()

        # if terminations == {}:
        #     terminations = {'prisoner': True, 'guard': True}
        # if truncations == {}:
        #     truncations = {'prisoner': True, 'guard': True}
        # self.prisoner_location = np.array([self.prisoner_x, self.prisoner_y])
        # self.guard_location = np.array([self.guard_x, self.guard_y])
        # self.escape_location = np.array([self.escape_x, self.escape_y])

        if self.render_mode == "human":
            self.render()

        # if prisoner_action == 4:
        #     print(prisoner_reward)
        # if guard_action == 4:
        #     print(guard_reward)

        # print(rewards)

        return observations, rewards, terminations, truncations, infos

    # def render(self):
    #     """Renders the environment."""
    #     grid = np.full((7, 7), " ")
    #     grid[self.prisoner_y, self.prisoner_x] = "P"
    #     grid[self.guard_y, self.guard_x] = "G"
    #     grid[self.escape_y, self.escape_x] = "E"
    #     print(f"{grid} \n")

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw obstacles
        # pix_square_size = self.window_size / self.size
        # for obstacle in self.obstacles:
        #     pygame.draw.rect(
        #         canvas,
        #         (128, 128, 128),  # Gray color for obstacles
        #         pygame.Rect(
        #             pix_square_size * np.array(obstacle),
        #             (pix_square_size, pix_square_size),
        #             ),
        #     )

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.escape_location,
                (pix_square_size, pix_square_size),
                ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.guard_location + 0.5) * pix_square_size,
            pix_square_size / 3,
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (54, 30, 54),
            (self.prisoner_location + 0.5) * pix_square_size,
            pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
            # rgb_array = np.transpose(
            #     np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            # )
            # # print("Shape of RGB array:", rgb_array.shape)
            # return rgb_array
        # rgb_array
        rgb_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        # print("Shape of RGB array:", rgb_array.shape)
        return rgb_array

    def get_obs(self):
        # grid = np.full((self.size, self.size), 0)
        # grid[self.prisoner_y, self.prisoner_x] = 1
        # grid[self.guard_y, self.guard_x] = 2
        # grid[self.escape_y, self.escape_x] = 3
        # obs = np.expand_dims(grid, axis=-1)
        # obs = self.render()

        # obs = (
        #     self.prisoner_x + 7 * self.prisoner_y,
        #     self.guard_x + 7 * self.guard_y,
        #     self.escape_x + 7 * self.escape_y,
        # )
        obs1 = (
            self.prisoner_location,
            self.guard_location,
            self.escape_location,
            np.array([0,0])
        )
        obs2 = (
            self.prisoner_location,
            self.guard_location,
            self.escape_location,
            np.array([1,1])
        )

        # result = {
        #     a: obs
        #     for a in self.agents
        # }
        result = {
            "prisoner": obs1,
            "guard": obs2,
        }


        # grid = np.full((7, 7), 0)
        # grid[self.prisoner_y, self.prisoner_x] = 1
        # grid[self.guard_y, self.guard_x] = 2
        # grid[self.escape_y, self.escape_x] = 3
        # result = np.expand_dims(grid, axis=-1)
        # print(result)


        return result


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # return MultiDiscrete([7 * 7] * 3)
        return Box(low=0, high=self.size-1, shape=(4,2), dtype=np.int32)
        # observation_space = spaces.Box(
        #     low=0, high=255, shape=(self.size, self.size, 1), dtype="uint8"
        # )
        # observation_space = spaces.Box(
        #     low=0, high=255, shape=(self.window_size, self.window_size, 3), dtype="uint8"
        # )
        # return observation_space

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)