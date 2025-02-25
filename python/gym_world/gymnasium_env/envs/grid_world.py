import copy
import sys
import os
from enum import Enum
from typing import List, Optional

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
import json

from gymnasium.utils import seeding
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import heapq
from sb3_contrib.common.maskable.utils import get_action_masks


def debug_print(*args):
    """Prints debug information to stderr."""
    print(" | ".join(map(str, args)), file=sys.stderr)



class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    # still = 4

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10):
        self.size = size  # The size of the square grid
        # self.window_size = 512 # The size of the PyGame window
        self.window_size = 512 # The size of the PyGame window
        # matrix = [[1 for _ in range(size)] for _ in range(size)]
        # self.grid = Grid(matrix=matrix)
        self._agent_location = None
        self._target_location = None
        self.obstacle_index = 0
        self.num_obstacles = 15
        self.num_patterns = 10
        self.width = size
        self.height = size
        self.max_steps = 100
        self.reward_range = (0, 1)

        # try:
        #     with open("obstacles.json", "r") as file:
        #         obstacles = json.load(file)
        #         self.obstacles = list(map(tuple, obstacles))
        # except FileNotFoundError:
        #     print("The file 'obstacles.json' does not exist.")
        # except json.JSONDecodeError:
        #     print("Error: The file 'obstacles.json' contains invalid JSON.")

        # matrix = [[1 for _ in range(self.size)] for _ in range(self.size)]
        # self.grid = Grid(matrix=matrix)
        # for obstacle in self.obstacles:
        #     self.grid.node(obstacle[0], obstacle[1]).walkable = False


        self.view_size = self.size // 2
        # self.view_size = 1
        total_view_size = self.view_size*2 + 1
        # total_view_size = self.size + 1
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            # shape=(total_view_size, total_view_size, 1),
            shape=(self.size, self.size, 1),
            # shape=(self.size, self.size),
            dtype="uint8",
        )
        # image_observation_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self.window_size, self.window_size, 3),
        #     dtype="uint8",
        # )
        # self.observation_space = spaces.Dict(
        #     {
        #         "image": image_observation_space,
        #         # "direction": spaces.Discrete(4),
        #         # "mission": mission_space,
        #     }
        # )
        self.observation_space = image_observation_space




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
            # Actions.still.value: np.array([0, 0]),
        }

        # self.obstacles = self._generate_obstacles()



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

        # Return a dictionary observation
        # result = {
        #     "agent": self._agent_location,
        #     "target": self._target_location,
        #     "obstacles": np.array(self.obstacles).flatten(),
        #     # "view": np.array(self.get_agent_view())
        #     # "grid": np.array(self.maze)
        # }
        # view = self.get_agent_view()
        self.get_maze()
        view = self.maze
        view = np.array(view, dtype="uint8")
        view = np.expand_dims(view, axis=-1)
        # view = self._render_frame()

        # result = {
        #     "image": view
        # }
        result = view
        # print(result)
        # print(result.shape)
        # print(f"\r{result}%", end='', flush=True)


        return result

    def get_maze(self):
        self.maze = np.zeros((self.size, self.size), dtype=int)
        for x, y in self.obstacles:
            self.maze[y, x] = 1
        x, y = self._target_location
        self.maze[y, x] = 4
        x, y = self._agent_location
        self.maze[y, x] = 3

    def get_agent_view(self):
        agent_x, agent_y = self._agent_location

        # Initialize the environment grid (1 = wall, 0 = free space)
        # Assuming self.grid is your environment grid
        self.get_maze()
        grid = self.maze  # e.g., a 2D numpy array


        # Calculate bounds for the 5x5 view
        min_x = max(agent_x - self.view_size, 0)
        max_x = min(agent_x + self.view_size + 1, grid.shape[0])
        min_y = max(agent_y - self.view_size, 0)
        max_y = min(agent_y + self.view_size + 1, grid.shape[1])

        # Extract the 5x5 view
        view = grid[min_x:max_x, min_y:max_y]

        # Pad the view to ensure it's 5x5
        pad_top = max(0, self.view_size - agent_x)
        pad_bottom = max(0, (agent_x + self.view_size + 1) - grid.shape[0])
        pad_left = max(0, self.view_size - agent_y)
        pad_right = max(0, (agent_y + self.view_size + 1) - grid.shape[1])

        view = np.pad(view, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=2)  # 2 = wall for padding
        # Set the center of the view (agent's position) to 0
        # view[half_view, half_view] = 0

        return view



    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def action_masks(self) -> np.ndarray:
        # Initialize action mask (0: invalid, 1: valid) for 5 actions
        action_mask = [1] * 4

        # Get agent's current position
        x, y = self._agent_location

        # Check boundaries and obstacles for each action
        if y == 0 or (x, y - 1) in self.obstacles:  # Up
            action_mask[Actions.up.value] = 0
        if y == self.size - 1 or (x, y + 1) in self.obstacles:  # Down
            action_mask[Actions.down.value] = 0
        if x == 0 or (x - 1, y) in self.obstacles:  # Left
            action_mask[Actions.left.value] = 0
        if x == self.size - 1 or (x + 1, y) in self.obstacles:  # Right
            action_mask[Actions.right.value] = 0

        # print(action_mask)

        # The Still action is always valid
        # action_mask[Actions.still.value] = True

        # Return the info dictionary with the action mask
        action_mask = np.array(action_mask)
        return action_mask

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.step_count = 0
        self._gen_grid()

        observation = self._get_obs()
        info = self._get_info()
        # info = self.valid_action_mask()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        self.step_count += 1

        terminated = False
        info = self._get_info()
        # print(action)
        # info = self.valid_action_mask()

        # if np.random.rand() < 0.5:
        #     target_action = self.action_space.sample()
        #     target_direction = self._action_to_direction[target_action]
        #     new_target_location = np.clip(
        #         self._target_location + target_direction, 0, self.size - 1
        #     )
        #     if tuple(new_target_location) not in self.obstacles:
        #         self._target_location = new_target_location


        path = self.astar(tuple(self._agent_location), tuple(self._target_location))
        distanceBefore = len(path)


        direction = self._action_to_direction[action]
        new_agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # if tuple(new_agent_location) not in self.obstacles:
        #     self._agent_location = new_agent_location
        self._agent_location = new_agent_location

        path = self.astar(tuple(self._agent_location), tuple(self._target_location))
        distanceAfter = len(path)

        if (distanceAfter < distanceBefore):
            reward = 1 / 100
        elif (distanceAfter > distanceBefore):
            reward = -2 / 100
        else:
            reward = -1 / 100

        # reward = 0


        # if tuple(new_agent_location) in self.obstacles: || distanceBefore == distanceAfter:
        #     reward = -1

        if np.array_equal(self._agent_location, self._target_location):
            terminated = True
            reward = 1 - 0.9 * (self.step_count / self.max_steps)

        if tuple(new_agent_location) in self.obstacles:
            terminated = True
            reward = 0



        # print(reward)

        observation = self._get_obs()

        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            reward = -1

        if self.render_mode == "human":
            self._render_frame()

        # print(self.step_count)

        return observation, reward, terminated, truncated, info

    def _gen_grid(self):
        file_path = r"C:\Users\User\Desktop\Personal Project\tankio-master\obstacle_patterns.json"
        # if not os.path.exists(file_path):
        #     self.generate_pattern()
        # #load the obstacle patterns from the file
        with open(file_path, 'r') as f:
            obstacle_patterns = json.load(f)

        selected_pattern = obstacle_patterns[self.obstacle_index]
        self.obstacles = set([tuple(coord) for coord in selected_pattern])
        for i in range(self.size):
            self.obstacles.add((i, 0))  # Left edge
            self.obstacles.add((i, self.size - 1))  # Right edge
            self.obstacles.add((0, i))  # Top edge
            self.obstacles.add((self.size - 1, i))  # Bottom edge

        path = None
        while path == None:
            # Generate a random position for the agent within the grid bounds
            # Generate a random position for the agent
            while True:
                agent_x = self.np_random.integers(1, self.width - 1)
                agent_y = self.np_random.integers(1, self.height - 1)
                if (agent_x, agent_y) not in self.obstacles:
                    break

            self._agent_location = np.array((agent_x, agent_y))

            # Generate a random position for the goal
            while True:
                goal_x = self.np_random.integers(1, self.width - 1)
                goal_y = self.np_random.integers(1, self.height - 1)
                if (goal_x, goal_y) not in self.obstacles and (goal_x, goal_y) != tuple(self._agent_location):
                    break

            self._target_location = np.array((goal_x, goal_y))

            path = self.astar(tuple(self._agent_location), tuple(self._target_location))

        # Update index to cycle through patterns
        # num_patterns = len(obstacle_patterns)
        num_patterns = 3
        self.obstacle_index = (self.obstacle_index + 1) % num_patterns


    def generate_pattern(self):
        # Generate 10 random obstacle patterns and apply discard rules
        obstacle_patterns = []
        for _ in range(self.num_patterns):
            pattern = set()
            while len(pattern) < self.num_obstacles:
                modifier = 3
                modifier_disc = modifier + 1
                x = int(self.np_random.integers(modifier, self.width - modifier))
                y = int(self.np_random.integers(modifier, self.height - modifier))
                if (x, y) not in pattern:
                    pattern.add((x, y))  # Ensure no duplicates

                # Apply discard rules during generation
                # for x in range(modifier_disc, self.width - modifier_disc):
                #     pattern.discard((x, modifier_disc))  # Remove obstacles along the vertical line
                #
                # for y in range(modifier_disc, self.height - modifier_disc):
                #     pattern.discard((modifier_disc, y))  # Remove obstacles along the horizontal line

            obstacle_patterns.append(list(pattern))  # Store as list for JSON compatibility

        # Save to a JSON file
        print(f"Generated {obstacle_patterns} patterns.")

        try:
            with open('obstacle_patterns.json', 'w') as f:
                json.dump(obstacle_patterns, f)
            print("Patterns saved successfully.")
        except Exception as e:
            print(f"Error saving patterns: {e}")


    def astar(self, start, goal):
        """
        Compute the shortest path using A* algorithm, without diagonal movement.
        """
        def heuristic(a, b):
            # Manhattan distance as the heuristic (since we can't move diagonally)
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_neighbors(pos):
            # Get the valid neighbors around the current position (only horizontal/vertical)
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (pos[0] + dx, pos[1] + dy)
                # neighbor = (
                #     max(0, min(pos[0] + dx, self.size - 1)),
                #     max(0, min(pos[1] + dy, self.size - 1))
                # )
                #     new_target_location = np.clip(
                #         self._target_location + target_direction, 0, self.size - 1
                #     )
                if neighbor not in self.obstacle_set \
                    and (0 <= neighbor[0] < self.size) \
                    and (0 <= neighbor[1] < self.size):
                    neighbors.append(neighbor)
            return neighbors

        # Priority queue (min-heap) for the open set
        open_list = []
        heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start))  # (f, g, pos)

        # Dictionary to store the best g value for each position
        g_cost = {start: 0}

        # Dictionary to store the parent of each node for path reconstruction
        came_from = {}

        self.obstacle_set = set(self.obstacles)

        while open_list:
            _, current_g, current = heapq.heappop(open_list)

            # If we reached the goal, reconstruct the path
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path  # Return the path from start to goal

            for neighbor in get_neighbors(current):
                # Calculate tentative g score for the neighbor
                tentative_g = current_g + 1  # All neighbors have a cost of 1 for grid movement
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current

        return None  # No path found


    def render(self):
        if self.render_mode == "rgb_array" or "human":
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
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

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
        line_width = 3
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=line_width,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=line_width,
            )

        # # Get the agent's view (for visible area)
        # agent_x, agent_y = self._agent_location
        # # Define the bounds of the visible area based on the agent's view
        # min_x = max(agent_x - self.view_size, 0)
        # max_x = min(agent_x + self.view_size + 1, self.size)
        # min_y = max(agent_y - self.view_size, 0)
        # max_y = min(agent_y + self.view_size + 1, self.size)
        # overlay = pygame.Surface((max_x * pix_square_size, max_y * pix_square_size))
        # # Fill the overlay with a gray color (with transparency)
        # overlay.fill((100,100,100))  # Semi-transparent gray
        # overlay.set_alpha(100)  # Semi-transparent gray
        # # Blit the visible overlay on top of the canvas
        # canvas.blit(overlay, (min_x * pix_square_size, min_y * pix_square_size))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])


        rgb_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        # print("Shape of RGB array:", rgb_array.shape)
        return rgb_array

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
