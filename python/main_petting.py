import pickle
import sys
import json
from typing import Optional

import numpy as np
# import pydevd_pycharm

import gymnasium as gym
import time
from matplotlib import pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import A2C, PPO, DQN
import os

from gym.envs.toy_text.frozen_lake import generate_random_map
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium_env
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import TimeLimit

# from agent import GridWorldAgent
from tqdm import tqdm

from petting_env.envs import CustomActionMaskedEnvironment
from pettingzoo.test import api_test, parallel_api_test


# try:
#     pydevd_pycharm.settrace('localhost', port=5678, stderrToServer=True, suspend=False)
# except ConnectionRefusedError:
#     pass

cell = None
path_grid_dimensions = None
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

def get_latest_model_and_iters(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if not model_files:
        return None

    # Sort by modification time, newest first
    model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model = model_files[0]

    return os.path.join(model_dir, latest_model)


def test_sb3(num_games: int = 3, render_mode: Optional[str] = None):

    env_fn = CustomActionMaskedEnvironment
    env = env_fn(render_mode=render_mode)
    api_test(env, num_cycles=1000)
    #env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False, map_name="8x8")

    # Load model
    model_dir = "models"
    latest_policy = get_latest_model_and_iters(model_dir)
    if latest_policy:
        model = PPO.load(latest_policy, device='cpu')
    else:
        debug_print("Policy not found.")
        exit(0)

    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            debug_print([targets[0]['cellX'], targets[0]['cellY']])
            env.guard1_location = np.array([agents[0]['cellX'], agents[0]['cellY']], dtype=int)
            env.guard2_location = np.array([agents[1]['cellX'], agents[1]['cellY']], dtype=int)
            env.prisoner_location = np.array([targets[0]['cellX'], targets[0]['cellY']], dtype=int)
            # print(obs)

            if termination or truncation:
                act = None
            else:
                act = model.predict(obs, deterministic=True)[0]
                act = act.item()

            debug_print(f"\nAgent: {agent}, \nObservation: \n{obs}, \nReward: {reward}, \nAction: {act}")
            env.step(act)
    env.close()

    # Run a test
    for _ in range(3):
        obs, info = env.reset()
        terminated = False
        while True:
            debug_print([targets[0]['cellX'], targets[0]['cellY']])
            action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
            #debug_print(action)
            env._agent_location = np.array([agents[0]['cellX'], agents[0]['cellY']], dtype=int)
            env._target_location = np.array([targets[0]['cellX'], targets[0]['cellY']], dtype=int)
            # obs, info = env.reset()
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
                # debug_print([targets[0]['cellX'], targets[0]['cellY']])
                # env._target_location = np.array([target['cellX'], target['cellY']], dtype=int)

            if agents and targets and not test_sb3_called:
                # print(agents)
                test_sb3_called = True  # Set the flag to True after calling

                env_fn = CustomActionMaskedEnvironment
                env = env_fn(render_mode='human')
                #env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False, map_name="8x8")

                # Load model
                model_dir = "models"
                latest_policy = get_latest_model_and_iters(model_dir)
                if latest_policy:
                    model = PPO.load(latest_policy, device='cpu')
                else:
                    debug_print("Policy not found.")
                    exit(0)

                # obs, info = env.reset()
                env.unwrapped.updateDrawables(agents=agents, target=targets[0])
                env.reset()


            if agents and targets:
                # debug_print("oof")
                # debug_print([targets[0]['cellX'], targets[0]['cellY']])
                env.unwrapped.updateDrawables(agents=agents, target=targets[0])
                # obs, info = env.reset()
                # obs = env.unwrapped._get_obs()
                # env._agent_location = np.array([agents[0]['cellX'], agents[0]['cellY']], dtype=int)
                # env._target_location = np.array([targets[0]['cellX'], targets[0]['cellY']], dtype=int)
                #obs, info = env.reset(agent=agents[0], target=targets[0])
                #print(targets[0])
                current_time = time.monotonic()
                if current_time - last_execution_time >= DELAY_INTERVAL:
                    for agentZoo in env.agent_iter(max_iter=env.num_agents):
                        # debug_print(agentZoo)
                        obs, reward, termination, truncation, info = env.last()
                        # print(obs)

                        if termination or truncation:
                            act = None
                        else:
                            act = model.predict(obs, deterministic=True)[0]
                            act = act.item()
                            # debug_print(act)

                        debug_print(f"\nAgent: {agentZoo}, Reward: {reward}, Action: {act}, Observation: \n{obs}")
                        env.step(act)

                        if (not termination) and agentZoo != "prisoner" and act is not None:
                            agent_direction = env.unwrapped.get_action_direction(act)
                            size = path_grid_dimensions['cols']
                            agent_number = int(agentZoo[len("guard"):])
                            agent_location = obs[agent_number]
                            agent = agents[agent_number - 1]
                            next_node = np.clip(
                                agent_location + agent_direction, 0, size - 1
                            )

                            if not np.all(next_node == agent_location):
                                target_x = next_node[0] * cell['width'] + cell['width']/2
                                target_y = next_node[1] * cell['height'] + cell['height']/2
                                # Calculate the direction vector from the agent's current position to the target cell's position
                                delta_x = target_x - (agent['topLeftX'] + agent['width'] / 2)
                                delta_y = target_y - (agent['topLeftY'] + agent['height'] / 2)

                                # Calculate the distance to the target
                                distance = abs(delta_x) + abs(delta_y)
                                # print(distance)

                                # Normalize the direction vector and apply speed
                                normalized_step_x = delta_x / distance if distance != 0 else 0
                                normalized_step_y = delta_y / distance if distance != 0 else 0
                                agent['topLeftX'] += normalized_step_x * agent['speed'] * 5
                                agent['topLeftY'] += normalized_step_y * agent['speed'] * 5
                                # agent['topLeftX'] = target_x
                                # agent['topLeftY'] = target_y





                    last_execution_time = current_time


                # debug_print(targets[0])


            # debug_print(agents[0]['topLeftX'],agents[0]['topLeftY'])
            print(json.dumps(agents), flush=True)  # Ensure it's flushed to stdout


    except json.JSONDecodeError:
        print("Error: Invalid JSON data received.", file=sys.stderr)


