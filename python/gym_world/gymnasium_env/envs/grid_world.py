import copy
import sys
from enum import Enum
from typing import List

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from numpy import random
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import heapq
from sb3_contrib.common.maskable.utils import get_action_masks


class MazeGenerator:
    def __init__(self, size):
        self.size = size
        self._agent_location = (0, 0)
        self._target_location = (self.size - 1, self.size - 1)
        self.obstacles = set()

    def generate_maze(self):
        while True:
            # Clear previous obstacles
            self.obstacles.clear()

            # Randomly place obstacles
            for _ in range(int(self.size**2 * 0.3)):  # Place obstacles in 30% of the grid
                x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
                if (x, y) != self._agent_location and (x, y) != self._target_location:
                    self.obstacles.add((x, y))

            # Check if a path exists from start to target
            path = astar(self.size, tuple(self._agent_location), tuple(self._target_location), tuple(self.obstacles))
            if path:
                return path  # Return path if a valid path is found

    def get_obstacles(self):
        return list(self.obstacles)

def astar(grid_size, start, target, obstacles):
    # Convert obstacles list to set for faster lookup
    obstacle_set = set(obstacles)

    # Initialize start and target nodes
    start_node = Node(start)
    target_node = Node(target)

    # Open list as a priority queue and closed set for visited nodes
    open_list = []
    closed_set = set()

    # Push start node to open list
    heapq.heappush(open_list, start_node)

    # Loop until open list is empty
    while open_list:
        # Get node with lowest f-cost
        current_node = heapq.heappop(open_list)

        # If we reached the target, reconstruct the path
        if current_node.position == target_node.position:
            path = []
            while current_node:
                path.append([int(current_node.position[0]), int(current_node.position[1])])
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        # Add current node to closed set
        closed_set.add(current_node.position)

        # Check each neighbor (up, down, left, right)
        for direction in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor_position = (current_node.position[0] + direction[0],
                                 current_node.position[1] + direction[1])

            # Check if the neighbor is within bounds and not an obstacle
            if (0 <= neighbor_position[0] < grid_size and
                    0 <= neighbor_position[1] < grid_size and
                    neighbor_position not in obstacle_set and
                    neighbor_position not in closed_set):

                # Create neighbor node
                neighbor_node = Node(neighbor_position, current_node)

                # Uniform cost: each move costs 1
                neighbor_node.g = current_node.g + 1

                # Manhattan heuristic: |x1 - x2| + |y1 - y2|
                neighbor_node.h = abs(neighbor_position[0] - target_node.position[0]) + \
                                  abs(neighbor_position[1] - target_node.position[1])

                # Total cost
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                # Check if neighbor is in open list with a lower f-cost
                if any(node.position == neighbor_node.position and node.f <= neighbor_node.f for node in open_list):
                    continue

                # Push neighbor to open list
                heapq.heappush(open_list, neighbor_node)

    return None  # Return None if no path is found

def debug_print(*args):
    """Prints debug information to stderr."""
    print(" | ".join(map(str, args)), file=sys.stderr)

class Node:
    def __init__(self, position, parent=None):
        self.position = position  # Position as (x, y) coordinates
        self.parent = parent      # Parent node in the path
        self.g = 0                # Distance from start node
        self.h = 0                # Heuristic distance to target
        self.f = 0                # Total cost

    def __lt__(self, other):
        return self.f < other.f   # Less-than comparison for heapq priority queue


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
        self.window_size = 512  # The size of the PyGame window
        matrix = [[1 for _ in range(size)] for _ in range(size)]
        self.grid = Grid(matrix=matrix)
        self._agent_location = None
        self._target_location = None
        # self.reward_range = (-1, 100)
        middle_row = size // 2
        walkable_column = size - 1  # Choose which cell in the middle row remains walkable

        # List to store obstacle coordinates
        # obstacles = []
        #
        # # Set the entire middle row as unwalkable except for one cell, and save obstacles
        # for col in range(size):
        #     if col != walkable_column:
        #         obstacles.append((middle_row, col))  # Save obstacle position
        #         self.grid.node(middle_row, col).walkable = False
        maze = MazeGenerator(size)
        maze.generate_maze()
        # obstacles = maze.get_obstacles()
        #obstacles = [(6,5),(5,2),(5,0),(6,2),(4, 0), (4, 9), (5, 1), (2, 2), (0, 14), (11, 14), (15, 5), (1, 15), (18, 10), (4, 2), (5, 3), (8, 2), (14, 15), (17, 14), (11, 7), (2, 4), (13, 1), (15, 7), (6, 4), (5, 5), (5, 14), (9, 12), (15, 0), (6, 6), (7, 5), (17, 9), (1, 3), (13, 5), (1, 12), (15, 11), (7, 7), (18, 16), (14, 3), (17, 2), (14, 12), (11, 4), (9, 16), (13, 7), (15, 4), (6, 1), (18, 0), (13, 16), (15, 13), (2, 13), (18, 18), (12, 17), (10, 1), (7, 2), (13, 18), (11, 18), (16, 14), (7, 11), (18, 11), (3, 16), (1, 0), (1, 9), (11, 11), (16, 16), (6, 17), (18, 8), (4, 17), (8, 8), (13, 4), (8, 17), (11, 13), (2, 10), (0, 13), (3, 2), (3, 11), (4, 10), (17, 13), (10, 7), (1, 4), (1, 13), (6, 3), (3, 4), (10, 0), (8, 12), (15, 8), (18, 13), (10, 2), (10, 11), (2, 16), (15, 10), (18, 15), (4, 16), (17, 10), (11, 3), (10, 13), (9, 15), (0, 12), (2, 9), (2, 18), (6, 0), (15, 12), (7, 8)]
        #obstacles = [(4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8)]
        # obstacles = [(3,3)]
        obstacles = [
            (0, 2), (9, 2),
            (1, 2), (1, 3), (1, 5), (1, 6),  # Row 1 obstacles
            (2, 1), (2, 3), (2, 6), (2, 8),  # Row 2 obstacles
            (3, 1), (3, 8),                  # Row 3 obstacles
            (4, 3), (4, 5), (4, 6),          # Row 4 obstacles
            (5, 1), (5, 3), (5, 6), (5, 8),  # Row 5 obstacles
            (6, 2), (6, 3), (6, 5), (6, 7),  # Row 6 obstacles
            (7, 4),                          # Row 7 obstacle
            (8, 2), (8, 3), (8, 6), (8, 7)   # Row 8 obstacles
        ]
        for obstacle in obstacles:
            self.grid.node(obstacle[0], obstacle[1]).walkable = False


        # Print saved obstacle coordinates
        print("Obstacles:", obstacles)
        self.obstacles = obstacles


        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Box(low=0, high=size - 1, shape=(4,), dtype=np.int32)
        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=np.array([self.size-1, self.size-1, self.size-1, self.size-1]),
        #     shape=(4,),
        #     dtype=np.int32
        # )
        self.observation_space = spaces.MultiBinary(2 * self.size * self.size)


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

    # def _generate_obstacles(self):
    #     """Randomly generate obstacle locations on the grid."""
    #     obstacles = set()
    #     while len(obstacles) < self.num_obstacles:
    #         obstacle = tuple(self.np_random.integers(0, self.size, size=2))
    #         obstacles.add(obstacle)
    #     return obstacles

    def _one_hot_encode(self, location, size):
        # Create a flat one-hot encoded vector for the grid
        one_hot = np.zeros(size * size, dtype=np.int32)
        index = location[0] * size + location[1]  # Calculate the flat index
        one_hot[index] = 1
        return one_hot

    def _get_obs(self):
        # obstacle_coords = np.array([tuple(obs) for obs in self.obstacles])
        # result = np.concatenate([self._agent_location, self._target_location])
        agent_one_hot = self._one_hot_encode(self._agent_location, self.size)
        target_one_hot = self._one_hot_encode(self._target_location, self.size)

        # Concatenate the one-hot encodings
        result = np.concatenate([agent_one_hot, target_one_hot])

        return result



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

        # self.obstacles = self._generate_obstacles()
        # for x, y in self.obstacles:
        #     self.grid.node(x, y).walkable = False

        # Choose the agent's location uniformly at random
        middle_column = self.size // 2
        # self._agent_location = np.array([0, 0])
        # self._target_location = np.array([2, 5])
        while True:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if tuple(self._agent_location) not in self.obstacles:
                break
        #
        # # Initialize target location without colliding with agent or obstacles
        while True:
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if (not np.array_equal(self._target_location, self._agent_location) and
                    tuple(self._target_location) not in self.obstacles):
                break
        # self._agent_location = np.array([0, 0])

        observation = self._get_obs()
        info = self._get_info()
        # info = self.valid_action_mask()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        info = self._get_info()
        # print(action)
        # info = self.valid_action_mask()

        # target_action = self.action_space.sample()
        # target_direction = self._action_to_direction[target_action]
        # new_target_location = np.clip(
        #     self._target_location + target_direction, 0, self.size - 1
        # )
        # if tuple(new_target_location) not in self.obstacles:
        #     self._target_location = new_target_location





        # grid_copy = copy.deepcopy(self.grid)
        # finder = AStarFinder()

        # start = grid_copy.node(self._agent_location[0], self._agent_location[1])
        # end = grid_copy.node(self._target_location[0], self._target_location[1])
        # path, runs = finder.find_path(start, end, grid_copy)

        path = astar(self.size, tuple(self._agent_location), tuple(self._target_location), tuple(self.obstacles))
        distanceBefore = len(path)

        # distanceBefore = np.linalg.norm(
        #     self._agent_location - self._target_location, ord=1
        # )
        # distanceBefore = len(path)
        # path_set = set((x, y) for x, y in path)

        direction = self._action_to_direction[action]
        new_agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if tuple(new_agent_location) not in self.obstacles:
            self._agent_location = new_agent_location
        # self._agent_location = new_agent_location

        # distanceAfter = np.linalg.norm(
        #     self._agent_location - self._target_location, ord=1
        # )


        # start = grid_copy.node(self._agent_location[0], self._agent_location[1])
        # end = grid_copy.node(self._target_location[0], self._target_location[1])
        # path, runs = finder.find_path(start, end, grid_copy)
        # distanceAfter = len(path)
        # if tuple(new_agent_location) == (path[0].x, path[0].y):
        #     # Agent stayed in the same cell
        #     distanceAfter = distanceBefore
        # elif tuple(new_agent_location) in path_set:
        #     # Agent moved to a new cell on the path
        #     distanceAfter = distanceBefore - 1
        # else:
        #     # Agent moved to a cell off the path
        #     distanceAfter = distanceBefore + 1
        path = astar(self.size, tuple(self._agent_location), tuple(self._target_location), tuple(self.obstacles))
        distanceAfter = len(path)



        terminated = np.array_equal(self._agent_location, self._target_location)

        #reward = (distanceBefore - distanceAfter + 1) / 2
        if terminated:
            reward = 1
            # print(reward)
        else:
            reward = (distanceBefore - distanceAfter) / 100
            # if tuple(new_agent_location) in self.obstacles: || distanceBefore == distanceAfter:
            #     reward = -1
            # if len(path_coordinates) > 1 and tuple(self._agent_location) == path_coordinates[1]:
            #     reward = 1
            #reward = -1
            if tuple(new_agent_location) in self.obstacles:
                reward = -1 / 100
                # print(reward, "obstacles")
                # print(action)
                # print(new_agent_location)

        print(reward)

        observation = self._get_obs()
        # print(observation)



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
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
