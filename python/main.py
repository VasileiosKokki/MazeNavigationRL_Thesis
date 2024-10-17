import pickle
import sys
import json

import numpy as np
import pydevd_pycharm

import gymnasium as gym
import time
from matplotlib import pyplot as plt
from stable_baselines3 import A2C, PPO, DQN
import os

from gym.envs.toy_text.frozen_lake import generate_random_map
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium_env
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import TimeLimit

# from agent import GridWorldAgent
from tqdm import tqdm


# try:
#     pydevd_pycharm.settrace('localhost', port=5678, stderrToServer=True, suspend=False)
# except ConnectionRefusedError:
#     pass

cell = None
agents = None
targets = None
env = None
model = None
obs = None
info = None
DELAY_INTERVAL = 0
last_execution_time = time.monotonic()  # Track the initial execution time



def debug_print(*args):
    """Prints debug information to stderr."""
    print(" | ".join(map(str, args)), file=sys.stderr)


def test_sb3(render=True):

    env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")
    #env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False, map_name="8x8")

    # Load model
    model = A2C.load('models/a2c_10000', env=env)

    # Run a test
    for _ in range(3):
        obs, info = env.reset(agent=agents[0], target=targets[0])
        terminated = False
        while True:
            debug_print([targets[0]['cellX'], targets[0]['cellY']])
            action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
            #debug_print(action)
            env._agent_location = np.array([agents[0]['cellX'], agents[0]['cellY']], dtype=int)
            env._target_location = np.array([targets[0]['cellX'], targets[0]['cellY']], dtype=int)
            obs, info = env.reset(agent=agents[0], target=targets[0])
            #print(targets[0])
            obs, _, terminated, _, _ = env.step(action.item())


            # if terminated:
            #     break

    env.close()

# if agents is not None:
#     test_sb3()
test_sb3_called = False
for line in sys.stdin:
    try:
        data = json.loads(line)
        if data.get('type') == 'oneTimeData':
            game_bounds_dimensions, path_grid_dimensions, unwalkable_cells = data.get('data').values()
            cell = {
                'width': game_bounds_dimensions['width'] /  path_grid_dimensions['cols'],
                'height': game_bounds_dimensions['height'] /  path_grid_dimensions['rows']
            }
        elif data.get('type') == 'drawables':
            drawables = data.get('data')
            agents = [drawable for drawable in drawables if drawable['type'] == 'agent']
            for agent in agents:
                # agent["topLeftX"] += 5
                startCell = {
                    "x": np.floor((agent['topLeftX'] + agent['width'] / 2) / cell['width']),
                    "y": np.floor((agent['topLeftY'] + agent['height'] / 2) / cell['height'])
                }
                agent['cellX'] = startCell['x']
                agent['cellY'] = startCell['y']

                # env._agent_location = np.array([agent['cellX'], agent['cellY']], dtype=int)

            targets = [drawable for drawable in drawables if drawable['type'] == 'player']
            for target in targets:
                endCell = {
                    "x": np.floor((target['topLeftX'] + target['width'] / 2) / cell['width']),
                    "y": np.floor((target['topLeftY'] + target['height'] / 2) / cell['height'])
                }
                target['cellX'] = endCell['x']
                target['cellY'] = endCell['y']
                debug_print([targets[0]['cellX'], targets[0]['cellY']])
                # env._target_location = np.array([target['cellX'], target['cellY']], dtype=int)

            if agents and targets and not test_sb3_called:
                # print(agents)
                test_sb3_called = True  # Set the flag to True after calling

                env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")
                #env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False, map_name="8x8")

                # Load model
                model = A2C.load('models/a2c_10000', env=env)

                obs, info = env.reset(agent=agents[0], target=targets[0])
                terminated = False

            if agents and targets:
                debug_print([targets[0]['cellX'], targets[0]['cellY']])
                action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
                #debug_print(action)
                # env._agent_location = np.array([agents[0]['cellX'], agents[0]['cellY']], dtype=int)
                # env._target_location = np.array([targets[0]['cellX'], targets[0]['cellY']], dtype=int)
                #obs, info = env.reset(agent=agents[0], target=targets[0])
                env.unwrapped.updateDrawables(agent=agents[0], target=targets[0])
                #print(targets[0])
                current_time = time.monotonic()
                if current_time - last_execution_time >= DELAY_INTERVAL:
                    obs, _, terminated, _, _ = env.step(action.item())

                    # agents[0]['topLeftX'] = obs["agent"][0] * cell['width'] + cell['width']/2
                    # agents[0]['topLeftY'] = obs["agent"][1] * cell['height'] + cell['height']/2
                    if (not terminated):
                        agent_direction = env.unwrapped.get_action_direction(action.item())
                        size = 20
                        agent_location = obs["agent"]
                        agent = agents[0]
                        next_node = np.clip(
                            agent_location + agent_direction, 0, size - 1
                        )

                        debug_print(next_node)
                        target_x = next_node[0] * cell['width'] + cell['width'] / 2
                        target_y = next_node[1] * cell['height'] + cell['height'] / 2

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


                    last_execution_time = current_time

                debug_print(targets[0])



            print(json.dumps(agents), flush=True)  # Ensure it's flushed to stdout


    except json.JSONDecodeError:
        print("Error: Invalid JSON data received.", file=sys.stderr)


