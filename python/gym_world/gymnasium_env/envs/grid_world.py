import sys
import os
from enum import Enum

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import json

import heapq


def debug_print(*args):
    """Prints debug information to stderr."""
    print(" | ".join(map(str, args)), file=sys.stderr)



class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, render_fps=4, size=10, num_obstacles=15, num_patterns=10, target_moving_pattern=0, dense_rewards=True, policy="CnnPolicy"):
        self.size = size  # The size of the square grid
        self.window_size = 512 # The size of the PyGame window
        self.policy = policy
        self._agent_location = None
        self._target_location = None
        self.target_moving_pattern = target_moving_pattern
        self.dense_rewards = dense_rewards
        self.obstacle_index = 0
        self.num_obstacles = num_obstacles
        self.num_patterns = num_patterns
        self.width = size
        self.height = size
        self.max_steps = 100
        self.reward_range = (-1, 1)
        self.metadata["render_fps"] = render_fps

        border_obstacle_count = 2 * self.size + 2 * (self.size - 2)  # = 4 * self.size - 4
        total_num_obstacles = self.num_obstacles + border_obstacle_count

        if self.policy == "CnnPolicy":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.size, self.size, 1),
                dtype="uint8"
            )
        elif self.policy == "MultiInputPolicy":
            self.observation_space = spaces.Dict({
                "agent_pos": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=np.int32),
                "target_pos": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=np.int32),
                "obstacles": spaces.Box(
                    low=0,
                    high=self.size - 1,
                    shape=(total_num_obstacles, 2),  # one (x, y) per obstacle
                    dtype="uint8"
                )
            })
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=self.size - 1,
                shape=(2 + 2 + 2 * total_num_obstacles,),
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

        if self.policy == "CnnPolicy":
            self.get_maze()
            view = self.maze
            view = np.array(view, dtype="uint8")
            view = np.expand_dims(view, axis=-1)
            result = view
        # elif self.policy == "MultiInputPolicy":
        #     result = {
        #         "agent": self._agent_location,
        #         "target": self._target_location,
        #         "obstacles": np.array(sorted(self.obstacles)).flatten(),
        #     }
        else:
            result = np.concatenate([
                np.array(self._agent_location),        # (x, y)
                np.array(self._target_location),       # (x, y)
                np.array(sorted(self.obstacles)).flatten(),    # [x1, y1, x2, y2, ...]
            ])

        result = result.astype(np.uint8)
        return result

    def get_maze(self):
        self.maze = np.zeros((self.size, self.size), dtype=int)
        for x, y in self.obstacles:
            self.maze[y, x] = 1
        x, y = self._target_location
        self.maze[y, x] = 4
        x, y = self._agent_location
        self.maze[y, x] = 3

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "wrong_steps": self.wrong_step_count
        }


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.step_count = 0
        self.wrong_step_count = 0
        self._gen_grid()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        self.step_count += 1

        terminated = False
        reward = 0

        if self.num_obstacles == 0:
            distance = self.manhattan(tuple(self._agent_location), tuple(self._target_location))
        else:
            path = self.astar(tuple(self._agent_location), tuple(self._target_location))
            distance = len(path)


        distanceBefore = distance

        direction = self._action_to_direction[action]
        new_agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if self.num_obstacles == 0:
            distance = self.manhattan(tuple(new_agent_location), tuple(self._target_location))
        else:
            path = self.astar(tuple(new_agent_location), tuple(self._target_location))
            distance = len(path)

        distanceAfter = distance

        if self.dense_rewards:
            if (distanceAfter < distanceBefore):
                reward = 1 / 100
            elif (distanceAfter > distanceBefore):
                reward = -2 / 100

        if (distanceAfter > distanceBefore):
            self.wrong_step_count += 1

        if self.target_moving_pattern == 1: # random
            if np.random.rand() < 0.8:
                target_action = self.action_space.sample()
                target_direction = self._action_to_direction[target_action]
                new_target_location = np.clip(
                    self._target_location + target_direction, 1, self.size - 2
                )
            else:
                new_target_location = self._target_location

            if np.array_equal(new_agent_location, self._target_location) and np.array_equal(new_target_location, self._agent_location):
                terminated = True
                reward = 1 - 0.9 * (self.step_count / self.max_steps)

            self._target_location = new_target_location

        if self.target_moving_pattern == 2: # moves further away
            distance_before = self.manhattan(tuple(self._agent_location), tuple(self._target_location))
            new_target_location = None

            if np.random.rand() < 0.8:
                for action in Actions:
                    target_action = action.value
                    target_direction = self._action_to_direction[target_action]
                    new_target_location = np.clip(
                        self._target_location + target_direction, 1, self.size - 2
                    )
                    distance_after = self.manhattan(tuple(self._agent_location), tuple(new_target_location))
                    if distance_after > distance_before:
                        break
            else:
                new_target_location = self._target_location


            if np.array_equal(new_agent_location, self._target_location) and np.array_equal(new_target_location, self._agent_location):
                terminated = True
                reward = 1 - 0.9 * (self.step_count / self.max_steps)

            self._target_location = new_target_location



        if np.array_equal(new_agent_location, self._target_location):
            terminated = True
            reward = 1 - 0.9 * (self.step_count / self.max_steps)

        if tuple(new_agent_location) in self.obstacles:
            terminated = True
            reward = 0
            self.wrong_step_count += 1


        self._agent_location = new_agent_location


        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            reward = -1

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _gen_grid(self):

        if self.num_patterns == 0:
            selected_pattern = self.generate_pattern() # always random pattern, not cycling between 10 patterns
        else:
            file_path = 'obstacle_patterns.json'
            if not os.path.exists(file_path):
                self.save_patterns()
            # load the obstacle patterns from the file
            with open(file_path, 'r') as f:
                obstacle_patterns = json.load(f)

            selected_pattern = obstacle_patterns[self.obstacle_index]

        self.obstacles = set([tuple(coord) for coord in selected_pattern])
        for i in range(self.size): # we add the borders
            self.obstacles.add((i, 0))  # Left edge
            self.obstacles.add((i, self.size - 1))  # Right edge
            self.obstacles.add((0, i))  # Top edge
            self.obstacles.add((self.size - 1, i))  # Bottom edge

        path = None
        while path == None:
            # Generate a random position for the agent within the grid bounds
            # Generate a random position for the agent
            while True:
                agent_x = self.np_random.integers(1, self.width - 2)
                agent_y = self.np_random.integers(1, self.height - 2)
                if (agent_x, agent_y) not in self.obstacles:
                    break

            self._agent_location = np.array((agent_x, agent_y))

            # Generate a random position for the goal
            while True:
                goal_x = self.np_random.integers(1, self.width - 2)
                goal_y = self.np_random.integers(1, self.height - 2)
                if (goal_x, goal_y) not in self.obstacles and (goal_x, goal_y) != tuple(self._agent_location):
                    break

            self._target_location = np.array((goal_x, goal_y))

            if self.num_obstacles == 0:
                path = self.manhattan(tuple(self._agent_location), tuple(self._target_location))
            else:
                path = self.astar(tuple(self._agent_location), tuple(self._target_location))

        # Update index to cycle through patterns
        if self.num_patterns != 0:
            self.obstacle_index = (self.obstacle_index + 1) % self.num_patterns


    def save_patterns(self):
        # Generate x random obstacle patterns and apply discard rules
        obstacle_patterns = []
        for _ in range(self.num_patterns):
            pattern = self.generate_pattern()
            obstacle_patterns.append(pattern)
        # Save to a JSON file
        print(f"Generated {obstacle_patterns} patterns.")

        try:
            with open('obstacle_patterns.json', 'w') as f:
                json.dump(obstacle_patterns, f)
            print("Patterns saved successfully.")
        except Exception as e:
            print(f"Error saving patterns: {e}")


    def generate_pattern(self):
        # Generate a single random obstacle pattern
        pattern = set()
        while len(pattern) < self.num_obstacles:
            modifier = 2
            x = int(self.np_random.integers(modifier, self.width - modifier - 1))
            y = int(self.np_random.integers(modifier, self.height - modifier - 1))
            pattern.add((x, y))
        return list(pattern)

    def manhattan(self, start, goal):
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def astar(self, start, goal):
        """
        Compute the shortest path using A* algorithm, without diagonal movement.
        """
        def heuristic(start, goal):
            # Manhattan distance as the heuristic (since we can't move diagonally)
            return self.manhattan(start, goal)

        def get_neighbors(pos):
            # Get the valid neighbors around the current position (only horizontal/vertical)
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (pos[0] + dx, pos[1] + dy)
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
