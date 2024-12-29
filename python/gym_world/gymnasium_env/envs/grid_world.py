import copy
import sys
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


class MazeGenerator:
    def __init__(self, size):
        self.width = size
        self.height = size
        self.size = size
        self.count = 0



    # DFS to check that it's a valid path.
    def is_valid(self, board: List[List[str]], max_size: int) -> bool:
        frontier, discovered = [], set()
        frontier.append(self._agent_location)
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                        continue
                    if board[r_new][c_new] == 3:
                        return True
                    if board[r_new][c_new] != 1:
                        frontier.append((r_new, c_new))
        return False

    def generate_random_map(
            self, size: int = 8, p: float = 0.8, seed: Optional[int] = None, num_obstacles = 0
    ) -> List[str]:
        """Generates a random valid map (one that has a path from start to goal)

        Args:
            size: size of each side of the grid
            p: probability that a tile is frozen
            seed: optional seed to ensure the generation of reproducible maps

        Returns:
            A random valid map
        """
        valid = False
        self.board = []  # initialize to make pyright happy

        np_random, _ = seeding.np_random(seed)

        while not valid:

            self.board = list(np.zeros((size, size), dtype=int))
            # if self.count == 0:
            self.flat_indices = np.random.choice(size * size, num_obstacles, replace=False)
            for index in self.flat_indices:
                row, col = divmod(index, size)  # Convert flat index to 2D coordinates
                self.board[row][col] = 1

            # predefined_obstacles = [
            #     (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
            #     (1, 7), (2, 7), (3, 7), (4, 7),
            #     (5, 1), (5, 2), (5, 3), (5, 4),
            #     (6, 4), (6, 5), (6, 6), (6, 8),
            #     (7, 1), (7, 3), (7, 6), (7, 9),
            #     (8, 0), (8, 2), (8, 5), (8, 6),
            #     (9, 8), (9, 9),
            #     # (10, 0), (10, 1), (10, 3), (10, 4),
            #     # (11, 5), (11, 6), (11, 9), (11, 11),
            #     # (12, 2), (12, 4), (12, 6), (12, 7),
            #     # (13, 2), (13, 5), (13, 8), (13, 11),
            #     # (14, 3), (14, 6), (14, 7), (14, 12)
            # ]
            # for x,y in predefined_obstacles:
            #     self.board[x][y] = 1

            while True:
                self._agent_location = np_random.integers(0, self.size, size=2, dtype=int)
                if self.board[self._agent_location[0]][self._agent_location[1]] == 0:
                    self.board[self._agent_location[0]][self._agent_location[1]] = 2
                    break
            while True:
                self._target_location = np_random.integers(0, self.size, size=2, dtype=int)
                if self.board[self._target_location[0]][self._target_location[1]] == 0:
                    self.board[self._target_location[0]][self._target_location[1]] = 3
                    break
            valid = self.is_valid(self.board, size)
            self.count += 1
        return ["".join(str(num) for num in row) for row in self.board]

    def get_obstacle_coordinates(self):
        # Get coordinates of all obstacles (cells with value 1)
        obstacles = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 1:
                    obstacles.append((x, y))

        return obstacles

    def get_agent_target_location(self):
        agent_location=None
        target_location=None
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 2:
                    agent_location=(x,y)
                if self.board[y][x] == 3:
                    target_location=(x,y)


        return np.array(agent_location),np.array(target_location)

    def get_maze_grid(self):
        temp_board = copy.deepcopy(self.board)
        # for y in range(self.height):
        #     for x in range(self.width):
        #         if temp_board[y][x] == 2:
        #             temp_board[y][x] = 0
        #         if temp_board[y][x] == 3:
        #             temp_board[y][x] = 0
        return temp_board

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

    def __init__(self, render_mode=None, size=7):
        self.size = size  # The size of the square grid
        # self.window_size = 512 # The size of the PyGame window
        self.window_size = 512 # The size of the PyGame window
        # matrix = [[1 for _ in range(size)] for _ in range(size)]
        # self.grid = Grid(matrix=matrix)
        self._agent_location = None
        self._target_location = None
        self.maze_generator = MazeGenerator(self.size)
        # self.reward_range = (0, 1)

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
        # self.observation_space = spaces.MultiBinary(2 * self.size * self.size)
        self.num_obstacles = 15
        # self.observation_space = spaces.Dict({
        #     "agent": spaces.MultiDiscrete([size, size]),
        #     "target": spaces.MultiDiscrete([size, size]),
        #     # "obstacles": spaces.MultiDiscrete([size, size] * 15)
        #     # "view": spaces.MultiBinary([5,5])
        #     # "grid": spaces.MultiBinary([size, size])
        # })
        # self.observation_space = spaces.MultiDiscrete(size * size * [4])
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(self.size, self.size, 4), dtype=np.uint8
        # )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.size, self.size, 1), dtype="uint8"
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



    def _get_obs(self):
        # obstacle_coords = np.array([tuple(obs) for obs in self.obstacles])
        # result = np.concatenate([self._agent_location, self._target_location])
        # Create the obstacle grid as a binary matrix
        # obstacles = self.get_closest_obstacles()
        # obstacles = np.array(self.obstacles).flatten()

        # Return a dictionary observation
        # result = {
        #     "agent": self._agent_location,
        #     "target": self._target_location,
        #     # "obstacles": obstacles,
        #     # "view": np.array(self.get_agent_view())
        #     # "grid": np.array(self.maze)
        # }

        # result = np.array(self.maze)
        # print(np.vstack(self.maze))
        # result = np.vstack(self.maze)

        self.board = list(np.zeros((self.size, self.size), dtype=int))
        for obstacle in self.obstacles:
            self.board[obstacle[0]][obstacle[1]] = 1
        if self.board[self._agent_location[0]][self._agent_location[1]] == 0:
            self.board[self._agent_location[0]][self._agent_location[1]] = 2

        if self.board[self._target_location[0]][self._target_location[1]] == 0:
            self.board[self._target_location[0]][self._target_location[1]] = 3

        print(np.array(self.board))
        result = np.expand_dims(np.array(self.board, dtype="uint8"), axis=-1)
        # print(result)
        # print("Observation:", result)
        # print("Observation Space:", self.observation_space)
        # print("Obstacles Shape:", np.array(self.obstacles).shape)

        # result = self.render()
        # print(result.shape)


        return result

    def get_closest_obstacles(self):
        agent_x, agent_y = self._agent_location

        # Calculate Manhattan distances to each obstacle
        distances = []
        for obstacle in self.obstacles:
            obstacle_x, obstacle_y = obstacle
            manhattan_distance = abs(agent_x - obstacle_x) + abs(agent_y - obstacle_y)
            distances.append((manhattan_distance, obstacle))

        # Sort obstacles by Manhattan distance
        distances.sort(key=lambda x: x[0])

        # Get the 10 closest obstacles
        closest_obstacles = [obstacle for _, obstacle in distances[:15]]

        return np.array(closest_obstacles).flatten()  # Flatten for output

    def get_agent_view(self):
        agent_x, agent_y = self._agent_location
        view_size = 5
        half_view = view_size // 2

        # Initialize the environment grid (1 = wall, 0 = free space)
        # Assuming self.grid is your environment grid
        self.maze = np.zeros((self.size, self.size), dtype=int)
        for x, y in self.obstacles:
            self.maze[y, x] = 1
        grid = self.maze  # e.g., a 2D numpy array


        # Calculate bounds for the 5x5 view
        min_x = max(agent_x - half_view, 0)
        max_x = min(agent_x + half_view + 1, grid.shape[0])
        min_y = max(agent_y - half_view, 0)
        max_y = min(agent_y + half_view + 1, grid.shape[1])

        # Extract the 5x5 view
        view = grid[min_x:max_x, min_y:max_y]

        # Pad the view to ensure it's 5x5
        pad_top = max(0, half_view - agent_x)
        pad_bottom = max(0, (agent_x + half_view + 1) - grid.shape[0])
        pad_left = max(0, half_view - agent_y)
        pad_right = max(0, (agent_y + half_view + 1) - grid.shape[1])

        view = np.pad(view, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=1)  # 1 = wall for padding

        # Set the center of the view (agent's position) to 0
        view[half_view, half_view] = 0

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

        maze = self.maze_generator
        maze.generate_random_map(size=self.size,num_obstacles=self.num_obstacles)

        self.obstacles = maze.get_obstacle_coordinates()
        # print(self.obstacles)
        self.maze = maze.get_maze_grid()
        # print(np.array(self.maze))
        matrix = [[1 for _ in range(self.size)] for _ in range(self.size)]
        self.grid = Grid(matrix=matrix)
        for obstacle in self.obstacles:
            self.grid.node(obstacle[0], obstacle[1]).walkable = False

        self._agent_location, self._target_location = maze.get_agent_target_location()
        # print(len(self.obstacles))

        # Choose the agent's location uniformly at random
        # middle_column = self.size // 2
        # self._agent_location = np.array([0, 0])
        # self._target_location = np.array([2, 5])
        # while True:
        #     self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        #     if tuple(self._agent_location) not in self.obstacles:
        #         break
        # while True:
        #     self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        #     if (not np.array_equal(self._target_location, self._agent_location) and
        #             tuple(self._target_location) not in self.obstacles):
        #         break

        # self._agent_location = np.array([5, 7])

        # self._target_location = np.array([3, 6])
        #
        # for _ in range(4):
        target_action = self.action_space.sample()
        #     target_direction = self._action_to_direction[target_action]
        #     new_target_location = np.clip(
        #         self._target_location + target_direction, 0, self.size - 1
        #     )
        #     if tuple(new_target_location) not in self.obstacles:
        #         self._target_location = new_target_location

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

        # if np.random.rand() < 0.5:
        #     target_action = self.action_space.sample()
        #     target_direction = self._action_to_direction[target_action]
        #     new_target_location = np.clip(
        #         self._target_location + target_direction, 0, self.size - 1
        #     )
        #     if tuple(new_target_location) not in self.obstacles:
        #         self._target_location = new_target_location





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

        # print(reward)

        observation = self._get_obs()



        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

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
        else:  # rgb_array
            rgb_array = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            # print("Shape of RGB array:", rgb_array.shape)
            return rgb_array

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
