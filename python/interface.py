import contextlib
import sys
import json
import warnings

import numpy as np

import gymnasium as gym
import gymnasium_env
import time

from stable_baselines3 import PPO
import os

import utils
from gymnasium.utils import seeding


cell = None
path_grid_dimensions = None
agents = None
targets = None
env = None
model = None
obs = None
info = None
np_random = None
np_random_seed = None
DELAY_INTERVAL = 0
last_execution_time = time.monotonic()  # Track the initial execution time

_real_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
# sys.stderr = open(os.devnull, "w")
warnings.filterwarnings("ignore")



def debug_print(*args):
    """Prints debug information to stderr."""
    print(" | ".join(map(str, args)), file=sys.stderr)

def json_print(data):
    """Always print JSON to the real stdout, even if stdout is redirected/suppressed."""
    print(json.dumps(data), flush=True, file=_real_stdout)



def reset_positions(agent, target, seed=None):
    global np_random, np_random_seed

    if seed is not None:
        np_random, np_random_seed = seeding.np_random(seed)

    size = path_grid_dimensions['cols']
    obstacles = get_obstacles(size)

    width = size
    height = size

    while True:
        agent_x = np_random.integers(1, width - 2)
        agent_y = np_random.integers(1, height - 2)
        if (agent_x, agent_y) not in obstacles:
            break

    agent_location = np.array((agent_x, agent_y))

    # Generate a random position for the goal
    while True:
        goal_x = np_random.integers(1, width - 2)
        goal_y = np_random.integers(1, height - 2)
        if (goal_x, goal_y) not in obstacles and (goal_x, goal_y) != tuple(agent_location):
            break

    target_location = np.array((goal_x, goal_y))

    agent['topLeftX'] = agent_location[0] * cell['width'] + cell['width']/2 - agent['width']/2
    agent['topLeftY'] = agent_location[1] * cell['height'] + cell['height']/2 - agent['height']/2

    target['topLeftX'] = target_location[0] * cell['width'] + cell['width']/2 - target['width']/2
    target['topLeftY'] = target_location[1] * cell['height'] + cell['height']/2 - target['height']/2


def get_obstacles(size):
    obstacles = []
    for i in range(size):
        # Top row
        obstacles.append([0, i])
        # Bottom row
        obstacles.append([size - 1, i])
        # Left column
        obstacles.append([i, 0])
        # Right column
        obstacles.append([i, size - 1])

    return obstacles

def update_drawable_cell(drawable, cell):
    """
    Updates the agent's cellX and cellY based on its topLeftX, topLeftY, width, and height.
    """
    startCell = {
        "x": np.floor((drawable['topLeftX'] + drawable['width'] / 2) / cell['width']),
        "y": np.floor((drawable['topLeftY'] + drawable['height'] / 2) / cell['height'])
    }
    drawable['cellX'] = startCell['x']
    drawable['cellY'] = startCell['y']

env_created = False
rejoinedPlayer = False
eval_mode = False
visited_cells = []
i = 0
episodes_desired_num = 100
for line in sys.stdin:
    try:
        data = json.loads(line)
        if data.get('type') == 'oneTimeData':
            game_bounds_dimensions, path_grid_dimensions, unwalkable_cells, eval_mode = data.get('data').values()
            cell = {
                'width': game_bounds_dimensions['width'] / path_grid_dimensions['cols'],
                'height': game_bounds_dimensions['height'] / path_grid_dimensions['rows']
            }
        elif data.get('type') == 'drawables':
            drawables = data.get('data')
            agents = [drawable for drawable in drawables if drawable['type'] == 'agent']
            for agent in agents:
                update_drawable_cell(agent, cell)

            targets = [drawable for drawable in drawables if drawable['type'] == 'player']
            for target in targets:
                update_drawable_cell(target, cell)

            if agents and not env_created:
                env_created = True  # Set the flag to False after calling

                folder = 'experiment2'
                model_dir = os.path.join("models", folder)
                latest_model_path = utils.get_latest_model_path(model_dir)

                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    env = gym.make('gymnasium_env/RealWorld-v0', render_mode="human")

                # Load model
                model = PPO.load(f'{latest_model_path}', env=env)

                reset_positions(agent=agents[0], target=agents[0], seed=42)
                env.unwrapped.updateDrawables(agent=agents[0], target=agents[0])
                _, _ = env.reset()

            if agents and targets:
                env.unwrapped.updateDrawables(agent=agents[0], target=targets[0])
                obs = env.unwrapped._get_obs()
                action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
                current_time = time.monotonic()
                agent = agents[0]
                if current_time - last_execution_time >= DELAY_INTERVAL:
                    obs, _, terminated, _, _ = env.step(action.item())

                    if (not terminated and not rejoinedPlayer):
                        agent_direction = env.unwrapped.get_action_direction(action.item())
                        size = path_grid_dimensions['cols']
                        agent_location = obs[:2]
                        next_node = np.clip(
                            agent_location + agent_direction, 0, size - 1
                        )

                        target_x = next_node[0] * cell['width'] + cell['width']/2
                        target_y = next_node[1] * cell['height'] + cell['height']/2
                        # Calculate the direction vector from the agent's current position to the target cell's position
                        delta_x = target_x - (agent['topLeftX'] + agent['width'] / 2)
                        delta_y = target_y - (agent['topLeftY'] + agent['height'] / 2)

                        # Calculate the distance to the target
                        distance = abs(delta_x) + abs(delta_y)

                        # Normalize the direction vector and apply speed
                        normalized_step_x = delta_x / distance if distance != 0 else 0
                        normalized_step_y = delta_y / distance if distance != 0 else 0
                        agent['topLeftX'] += normalized_step_x * agent['speed'] * 5
                        agent['topLeftY'] += normalized_step_y * agent['speed'] * 5
                    elif eval_mode:
                        if rejoinedPlayer:
                            rejoinedPlayer = False
                            reset_positions(agent=agents[0], target=targets[0], seed=42)
                            update_drawable_cell(agents[0], cell)
                            update_drawable_cell(targets[0], cell)
                        else:
                            reset_positions(agent=agents[0], target=targets[0])

                    last_execution_time = current_time

                    if i <= episodes_desired_num - 1:
                        current_pos = (agent['cellX'], agent['cellY'])
                        utils.collect_agent_positions(current_pos, visited_cells, i)

                        if i == episodes_desired_num - 1:
                            utils.save_agent_positions(visited_cells, "live_experiment", path_grid_dimensions['cols'])

                    if terminated:
                        i += 1
            elif eval_mode:
                rejoinedPlayer = True

            json_print(agents)
            json_print(targets)



    except json.JSONDecodeError:
        print("Error: Invalid JSON data received.", file=sys.stderr)


