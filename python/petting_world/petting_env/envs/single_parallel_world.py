import functools
import heapq
import json
from copy import copy
from enum import Enum
from typing import Optional

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete, Dict, Tuple
from gymnasium.utils import seeding
from pettingzoo.utils import agent_selector
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils.env import ActionType, AgentID


class Actions(Enum):
    right = 1
    up = 2
    left = 0
    down = 3

class CustomActionMaskedEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_aec_v0",
        "render_fps": 1,
    }

    def __init__(self, render_mode=None, size=10):
        self.size = size
        self.window_size = 512

        # self.escape_location = None
        self.guard1_location = None
        self.guard2_location = None
        self.prisoner_location = None
        self.timestep = None
        self.possible_agents = ["guard1"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.obstacle_index = 0
        self.num_obstacles = 15
        self.num_patterns = 10
        self.width = size
        self.height = size
        self.max_steps = 100
        self.reward_range = (0, 1)


        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = None
        self.dones = {agent: False for agent in self.possible_agents}
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

        self._gen_grid()

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

        observations = self.get_all_obs()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        if self.render_mode == "human":
            self.render()

        return observations, infos

    def step(self, actions):
        previously_done = [
            agent for agent in self.agents
            if self.terminations.get(agent, False) or self.truncations.get(agent, False)
               and agent != self.agent_selection  # Don't remove current agent yet
        ]
        # print(actions)
        if "guard1" in actions :
            guard1_action = actions["guard1"]
        else:
            guard1_action = -1

        if "guard2" in actions:
            guard2_action = actions["guard2"]
        else:
            guard2_action = -1

        # new_prisoner_location = self.prisoner_location
        # if self.has_neighbours():
        #     max_attempts = 30  # Set a limit to prevent infinite loops
        #     attempts = 0  # Counter to track attempts
        #     while True:
        #         prisoner_action = self.action_space("guard1").sample()
        #         attempts += 1
        #         prisoner_direction = self._action_to_direction[prisoner_action]
        #
        #         new_prisoner_location = np.clip(
        #             self.prisoner_location + prisoner_direction, 0, self.size - 1
        #         )
        #         if not (np.array_equal(new_prisoner_location, self.guard1_location) or np.array_equal(new_prisoner_location, self.guard2_location)) and not np.array_equal(new_prisoner_location,self.prisoner_location):
        #             break
        #         if attempts==max_attempts:
        #             new_prisoner_location = self.prisoner_location
        #             print("no space")
        #             break

        # print(actions)
        # prisoner_action = self.action_space("guard1").sample()
        # prisoner_direction = self._action_to_direction[prisoner_action]
        # new_prisoner_location = np.clip(
        #     self.guard1_location + prisoner_direction, 0, self.size - 1
        # )
        new_prisoner_location = self.prisoner_location

        if guard1_action != -1 :
            guard1_direction = self._action_to_direction[guard1_action]
            new_guard1_location = np.clip(
                self.guard1_location + guard1_direction, 0, self.size - 1
            )
        else:
            new_guard1_location = self.guard1_location

        if guard2_action != -1 :
            guard2_direction = self._action_to_direction[guard2_action]
            new_guard2_location = np.clip(
                self.guard2_location + guard2_direction, 0, self.size - 1
            )
        else:
            new_guard2_location = self.guard2_location

        # guard1_reward = 0
        # guard2_reward = 0


        # self.prisoner_location = new_prisoner_location

        # distance1 = abs(self.guard1_location[0] - self.prisoner_location[0]) + abs(self.guard1_location[1] - self.prisoner_location[1])
        # distance2 = abs(self.guard2_location[0] - self.prisoner_location[0]) + abs(self.guard2_location[1] - self.prisoner_location[1])
        #
        # new_distance1 = abs(new_guard1_location[0] - self.prisoner_location[0]) + abs(new_guard1_location[1] - self.prisoner_location[1])
        # new_distance2 = abs(new_guard2_location[0] - self.prisoner_location[0]) + abs(new_guard2_location[1] - self.prisoner_location[1])
        path = self.astar(tuple(self.guard1_location), tuple(self.prisoner_location))
        distance1 = len(path)
        # path = self.astar(tuple(self.guard2_location), tuple(self.prisoner_location))
        # distance2 = len(path)

        new_path = self.astar(tuple(new_guard1_location), tuple(self.prisoner_location))
        new_distance1 = len(new_path)
        # new_path = self.astar(tuple(new_guard2_location), tuple(self.prisoner_location))
        # new_distance2 = len(new_path)

        if new_distance1 < distance1:
            guard1_reward = 1 / 100
        elif new_distance1 > distance1:
            guard1_reward = -2 / 100
        else:
            guard1_reward = -1 / 100

        # if new_distance2 < distance2:
        #     guard2_reward += 0.05
        # else:
        #     guard2_reward -= 0.1

        # if np.array_equal(new_guard1_location, new_guard2_location):
        #     guard1_reward -= 0.05
        #     guard2_reward -= 0.05


        # terminations = {a: False for a in self.agents}
        to_terminate1 = False
        # to_terminate2 = False
        if np.array_equal(new_prisoner_location, new_guard1_location):
            guard1_reward = 1
            # guard2_reward = 1
            # terminations = {a: True for a in self.agents}
            to_terminate1 = True
            # to_terminate2 = True
        # elif np.array_equal(self.prisoner_location, self.escape_location):
        #     prisoner_reward = 1
        #     guard1_reward = -1
        #     terminations = {a: True for a in self.agents}
        if tuple(new_guard1_location) in self.obstacles:
            guard1_reward = 0
            to_terminate1 = True
            # self._agent_selector.reinit(["guard_2"])

        # if tuple(new_guard2_location) in self.obstacles:
        #     guard2_reward = 0
        #     to_terminate2 = True
        #     # self._agent_selector.reinit(["guard_1"])

        # terminations = {"guard1": to_terminate1, "guard2": to_terminate2}
        terminations = {}
        for agent in self.agents:
            terminations[agent] = to_terminate1 if agent == "guard1" else to_terminate2

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            guard1_reward = -1
            # guard2_reward = -1
            truncations = {a: True for a in self.agents}
        self.timestep += 1

        # rewards = {"guard1": guard1_reward, "guard2": guard2_reward}
        rewards = {}
        for agent in self.agents:
            rewards[agent] = guard1_reward if agent == "guard1" else guard2_reward
        # Get observations
        observations = self.get_all_obs()

        # Get dummy infos (not used in this example)
        # infos = {a: {} for a in self.agents}
        infos = {
            a: {
                "agent_id": a,
                "reward": rewards[a],
                "done": terminations[a] or truncations[a],
            }
            for a in self.agents
        }

        # if any(terminations.values()) or all(truncations.values()):
        #     self.agents = []

        # done_agents = [agent for agent in self.agents
        #                if terminations.get(agent, False) or truncations.get(agent, False)]
        # self.agents = [agent for agent in self.agents if agent not in done_agents]
        # self.agents = [a for a in self.agents if a not in previously_done]
        self.truncations = truncations
        self.terminations = terminations

        self.agents = [
            agent
            for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]


        # Also remove them from all relevant dictionaries
        # for agent in previously_done:
        #     terminations.pop(agent, None)
        #     truncations.pop(agent, None)
        #     rewards.pop(agent, None)
        #     observations.pop(agent, None)
        #     infos.pop(agent, None)

        # print("Remaining agents:", self.agents)
        # print(rewards)


        self.prisoner_location = new_prisoner_location
        self.guard1_location = new_guard1_location
        # self.guard2_location = new_guard2_location

        if self.render_mode == "human":
            self.render()

        # print(observations, rewards, terminations, truncations, infos)

        return observations, rewards, terminations, truncations, infos

    def has_neighbours(self):
        neighbour = False
        for i in range(4):
            prisoner_action = i
            prisoner_direction = self._action_to_direction[prisoner_action]

            new_prisoner_location = np.clip(
                self.prisoner_location + prisoner_direction, 0, self.size - 1
            )
            if (np.array_equal(new_prisoner_location, self.guard1_location) or np.array_equal(new_prisoner_location, self.guard2_location)):
                neighbour = True
                break
        return neighbour

    def observe(self, agent):
        # xreiazetai mono gia to api test
        observation = self.get_all_obs()

        # Return the observation inside a dictionary with the correct structure
        return observation[agent]

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

        while True:
            # Generate a random position for the agent within the grid bounds
            # Generate a random position for the agent
            while True:
                agent_x = self.np_random.integers(1, self.width - 1)
                agent_y = self.np_random.integers(1, self.height - 1)
                if (agent_x, agent_y) not in self.obstacles:
                    break

            self.guard1_location = np.array((agent_x, agent_y))

            # while True:
            #     agent_x = self.np_random.integers(1, self.width - 1)
            #     agent_y = self.np_random.integers(1, self.height - 1)
            #     if (agent_x, agent_y) not in self.obstacles and (agent_x, agent_y) != tuple(self.guard1_location):
            #         break
            #
            # self.guard2_location = np.array((agent_x, agent_y))

            # print(self.obstacles)
            # print("-----",self.guard2_location,"===\n")

            # Generate a random position for the goal
            while True:
                goal_x = self.np_random.integers(1, self.width - 1)
                goal_y = self.np_random.integers(1, self.height - 1)
                if (goal_x, goal_y) not in self.obstacles and (goal_x, goal_y) != tuple(self.guard1_location):
                    break

            self.prisoner_location = np.array((goal_x, goal_y))

            path1 = self.astar(tuple(self.guard1_location), tuple(self.prisoner_location))
            # print(path1)
            # print(path2)
            if path1 != None:
                break


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

    def get_maze(self):
        maze = np.zeros((self.size, self.size), dtype=int)
        for x, y in self.obstacles:
            maze[y, x] = 1
        x, y = self.prisoner_location
        maze[y, x] = 2
        x, y = self.guard1_location
        maze[y, x] = 3
        # x, y = self.guard2_location
        # maze[y, x] = 4

        return maze

    def get_current_player_maze(self, agent):
        maze = np.zeros((self.size, self.size), dtype=int)
        if agent == "prisoner":
            x, y = self.prisoner_location
            maze[y, x] = 2
        if agent == "guard1":
            x, y = self.guard1_location
            maze[y, x] = 3
        if agent == "guard2":
            x, y = self.guard2_location
            maze[y, x] = 4


        return maze

    def get_obs(self, agent):

        view = self.get_maze()
        view = np.array(view, dtype="uint8")
        view = np.expand_dims(view, axis=-1)

        # # Get the current player's specific view (second channel)
        # current_player_view = self.get_current_player_maze(agent)
        # current_player_view = np.array(current_player_view, dtype="uint8")
        # current_player_view = np.expand_dims(current_player_view, axis=-1)  # Adding a channel dimension (shape: (size, size, 1))
        #
        # # Stack the two channels together (shape: (size, size, 2))
        # combined_view = np.concatenate([view, current_player_view], axis=-1)

        return view


    def get_all_obs(self):
        # observation = {
        #     "guard1": self.get_obs("guard1"),
        #     "guard2": self.get_obs("guard2"),
        # }
        observation = {}
        if "guard1" in self.agents:
            observation["guard1"] = self.get_obs("guard1")
        if "guard2" in self.agents:
            observation["guard2"] = self.get_obs("guard2")
        # print(observation)

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
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self.escape_location,
        #         (pix_square_size, pix_square_size),
        #         ),
        # )
        # Now we draw the agent
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (54, 30, 54),
            (self.prisoner_location + 0.5) * pix_square_size,
            pix_square_size / 3,
            )
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.guard1_location + 0.5) * pix_square_size,
            pix_square_size / 3,
            )
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 200),
        #     (self.guard2_location + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        #     )

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
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            # shape=(total_view_size, total_view_size, 1),
            shape=(self.size, self.size, 1),
            # shape=(self.size, self.size),
            dtype="uint8",
        )
        # return Box(low=0, high=self.size - 1, shape=(3, 2), dtype=np.int32)
        return image_observation_space



    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)
