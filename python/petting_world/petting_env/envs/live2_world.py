import functools
import sys
from copy import copy
from enum import Enum
from typing import Optional

import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete, Dict, Tuple
from gymnasium.utils import seeding
from pettingzoo.utils import agent_selector
from pettingzoo import AECEnv
from pettingzoo.utils.env import ActionType, AgentID


def debug_print(*args):
    """Prints debug information to stderr."""
    print(" | ".join(map(str, args)), file=sys.stderr)

class Actions(Enum):
    right = 1
    up = 2
    left = 0
    down = 3

class CustomActionMaskedEnvironment(AECEnv):
    metadata = {
        "name": "custom_environment_aec_v0",
        "render_fps": 60,
        "is_parallelizable": True
    }

    def __init__(self, render_mode=None):
        self.size = 15
        self.window_size = 512

        # self.escape_location = None
        self.guard1_location = None
        self.guard2_location = None
        self.prisoner_location = None
        self.timestep = None
        self.possible_agents = ["guard1", "guard2"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = None
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, -1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, 1]),
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.agents = copy(self.possible_agents)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._actions: Dict[AgentID, Optional[ActionType]] = {
            agent: None for agent in self.agents
        }

        self.timestep = 0
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}

        # self.prisoner_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        #
        # while True:
        #     self.guard1_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        #     if not np.array_equal(self.guard1_location, self.prisoner_location):
        #         break
        # while True:
        #     self.guard2_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        #     if not np.array_equal(self.guard2_location, self.prisoner_location):
        #         break

        # while True:
        #     self.escape_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        #     if not np.array_equal(self.escape_location, self.prisoner_location) and not np.array_equal(self.escape_location, self.guard1_location):
        #         break

        self.infos = {a: {} for a in self.agents}

    def step(self, action: Optional[ActionType]):
        # print("oof")
        # if (
        #         self.terminations[self.agent_selection]
        #         or self.truncations[self.agent_selection]w
        # ):
        #     del self._actions[self.agent_selection]
        #     assert action is None
        #     self._was_dead_step(action)
        #     return
        self._actions[self.agent_selection] = action
        if self._agent_selector.is_last():
            obss, rews, terminations, truncations, infos = self.step_all(self._actions)

            self._observations = copy(obss)
            self.terminations = copy(terminations)
            self.truncations = copy(truncations)
            self.infos = copy(infos)
            self.rewards = copy(rews)
            self._cumulative_rewards = copy(rews)

            env_agent_set = set(self.agents)

            self.agents = self.agents + [
                agent
                for agent in sorted(self._observations.keys(), key=lambda x: str(x))
                if agent not in env_agent_set
            ]

            if len(self.agents):
                self._agent_selector = agent_selector(self.agents)
                self.agent_selection = self._agent_selector.reset()

            # self._deads_step_first()
        else:
            if self._agent_selector.is_first():
                self._clear_rewards()

            self.agent_selection = self._agent_selector.next()

    def step_all(self, actions):
        # print(actions)
        if "guard1" in actions and "guard2" in actions:
            prisoner_action = 4
            guard1_action = actions["guard1"]
            guard2_action = actions["guard2"]
        else:
            prisoner_action = -1
            guard1_action = -1
            guard2_action = -1

        # prisoner_direction = self._action_to_direction[prisoner_action]
        # # print(prisoner_direction,"prisoner")
        # new_prisoner_location = np.clip(
        #     self.prisoner_location + prisoner_direction, 0, self.size - 1
        # )
        # guard1_direction = self._action_to_direction[guard1_action]
        # guard2_direction = self._action_to_direction[guard2_action]
        # new_guard1_location = np.clip(
        #     self.guard1_location + guard1_direction, 0, self.size - 1
        # )
        # new_guard2_location = np.clip(
        #     self.guard2_location + guard2_direction, 0, self.size - 1
        # )

        guard1_reward = 0
        guard2_reward = 0
        # dont stay still
        # if np.array_equal(self.prisoner_location, new_prisoner_location):
        #     prisoner_reward += -0.05
        # if np.array_equal(self.guard1_location, new_guard1_location):
        #     guard1_reward += -0.05
        # if np.array_equal(self.guard2_location, new_guard2_location):
        #     guard2_reward += -0.05
        #
        # self.prisoner_location = new_prisoner_location
        # self.guard1_location = new_guard1_location
        # self.guard2_location = new_guard2_location
        terminations = {a: False for a in self.agents}
        if np.array_equal(self.prisoner_location, self.guard1_location) or np.array_equal(self.prisoner_location, self.guard2_location):
            guard1_reward = 1
            guard2_reward = 1
            terminations = {a: True for a in self.agents}
        # elif np.array_equal(self.prisoner_location, self.escape_location):
        #     prisoner_reward = 1
        #     guard1_reward = -1
        #     terminations = {a: True for a in self.agents}
        guard1_reward += -0.1
        guard2_reward += -0.1

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        # if self.timestep > 100:
        #     prisoner_reward = 1
        #     guard1_reward = -1
        #     guard2_reward = -1
        #     truncations = {"prisoner": True, "guard1": True, "guard2": True}
        # self.timestep += 1

        rewards = {"guard1": guard1_reward, "guard2": guard2_reward}
        # Get observations
        observations = self.get_obs()

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        # if any(terminations.values()) or all(truncations.values()):
        #     self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def observe(self, agent):
        # xreiazetai mono gia to api test
        observation = self.get_obs()

        # Return the observation inside a dictionary with the correct structure
        return observation[agent]


    def get_obs(self):
        observation = {
            "guard1": np.array((self.prisoner_location, self.guard1_location, self.guard2_location, np.array([1,1]))),
            "guard2": np.array((self.prisoner_location, self.guard1_location, self.guard2_location, np.array([2,2]))),
        }

        return observation

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
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self.escape_location,
        #         (pix_square_size, pix_square_size),
        #         ),
        # )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.guard1_location + 0.5) * pix_square_size,
            pix_square_size / 3,
            )
        pygame.draw.circle(
            canvas,
            (0, 0, 200),
            (self.guard2_location + 0.5) * pix_square_size,
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

    def close(self):
        # Your close logic here
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=self.size - 1, shape=(4, 2), dtype=np.int32)



    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)


    def get_action_direction(self, action):
        return self._action_to_direction[action]

    def updateDrawables(self, agents=None, target=None):
        if agents is not None:
            self.guard1_location = np.array([agents[0]['cellX'], agents[0]['cellY']], dtype=int)
            self.guard2_location = np.array([agents[1]['cellX'], agents[1]['cellY']], dtype=int)
        if target is not None:
            self.prisoner_location = np.array([target['cellX'], target['cellY']], dtype=int)
            # debug_print(self.prisoner_location)
