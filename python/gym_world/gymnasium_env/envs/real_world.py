import sys
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

def debug_print(*args):
    """Prints debug information to stderr."""
    print(" | ".join(map(str, args)), file=sys.stderr)


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    # still = 4

class RealWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, size=10):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self._agent_location = None
        self._target_location = None

        border_obstacle_count = 2 * self.size + 2 * (self.size - 2)

        self.obstacles = set()
        for i in range(self.size): # we add the borders
            self.obstacles.add((i, 0))  # Left edge
            self.obstacles.add((i, self.size - 1))  # Right edge
            self.obstacles.add((0, i))  # Top edge
            self.obstacles.add((self.size - 1, i))  # Bottom edge

        self.observation_space = spaces.Box(
            low=0,
            high=self.size - 1,
            shape=(2 + 2 + 2 * border_obstacle_count,),
            dtype="uint8"
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, -1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, 1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        result = np.concatenate([np.array(self._agent_location), np.array(self._target_location), np.array(sorted(self.obstacles)).flatten(),])
        result = result.astype(np.uint8)
        return result


    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def get_action_direction(self, action):
        return self._action_to_direction[action]
    def updateDrawables(self, agent=None, target=None):
        if agent is not None:
            self._agent_location = np.array([agent['cellX'], agent['cellY']], dtype=int)
        if target is not None:
            self._target_location = np.array([target['cellX'], target['cellY']], dtype=int)


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action=None):

        info = self._get_info()

        terminated = np.array_equal(self._agent_location, self._target_location)


        if terminated:
            reward = 100
            # debug_print(reward)
        else:
            reward = 0


        observation = self._get_obs()


        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw obstacles
        pix_square_size = self.window_size / self.size
        for obstacle in self.obstacles:
            pygame.draw.rect(
                canvas,
                (128, 128, 128),  # Gray color for obstacles
                pygame.Rect(
                    pix_square_size * np.array(obstacle),
                    (pix_square_size, pix_square_size),
                    ),
            )

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
                ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
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
            # self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
