import pickle
import random
import re
import sys
import json
import time
import uuid

import numpy as np
import pydevd_pycharm

import gymnasium as gym
import torch
from matplotlib import pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import A2C, PPO, DQN
import os
import tensorflow as tf

from gym.envs.toy_text.frozen_lake import generate_random_map
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv, VecTransposeImage
from sb3_contrib import QRDQN
from sb3_contrib import RecurrentPPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from torch import nn

import gymnasium_env
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import TimeLimit

# from agent import GridWorldAgent
from tqdm import tqdm

from gymnasium_env.envs import GridWorldEnv
from gymnasium_env.envs.grid_world import MazeGenerator


class SaveOnTimestepCallback(BaseCallback):
    def __init__(self, model, save_path, save_interval, model_name):
        super(SaveOnTimestepCallback, self).__init__()
        self.model = model
        self.save_path = save_path
        self.save_interval = save_interval
        self.last_save = 0
        self.model_name = model_name

    def _on_step(self) -> bool:
        # Check if we reached the save interval
        if self.num_timesteps - self.last_save >= self.save_interval:
            self.last_save = self.num_timesteps
            checkpoint_path = os.path.join(self.save_path, f"{self.model_name}_{self.num_timesteps}.zip")
            self.model.save(checkpoint_path)
        return True

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=225):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # Should be 1 for your case (1, 15, 15)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )


        # Compute shape after CNN layers to define the FC layer size
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]


        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = self.cnn(observations)
        x = self.fc(x)
        return x



def get_latest_model_and_iters(model_dir, timesteps):
    """
    Find the latest model file and determine the current iteration number.
    Args:
        model_dir (str): Directory containing model files.
        timesteps (int): Number of timesteps per iteration.
    Returns:
        tuple: (path_to_latest_model, iteration_number)
    """
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if not model_files:
        return None, 0

    # Sort by modification time, newest first
    model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model = model_files[0]

    # Extract iteration number from the model filename (e.g., "model_100000.zip")
    match = re.search(r"_(\d+)\.zip$", latest_model)
    if match:
        iter_number = int(match.group(1)) // timesteps
    else:
        iter_number = 0  # Default to 0 if no match is found

    return os.path.join(model_dir, latest_model), iter_number

def run_q(episodes, is_training=True, render=False):

    env = gym.make("gymnasium_env/GridWorld-v0", render_mode='human' if render else None)
    env = gym.wrappers.RecordEpisodeStatistics(env,deque_size=episodes)

    if(is_training):
        # If training, initialize the Q Table, a 5D vector: [robot_row_pos, robot_row_col, target_row_pos, target_col_pos, actions]
        q = np.zeros((env.unwrapped.size, env.unwrapped.size, env.unwrapped.size, env.unwrapped.size, env.action_space.n))
    else:
        # If testing, load Q Table from file.
        f = open('v0_warehouse_solution.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # Hyperparameters
    learning_rate_a = 0.9   # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1             # 1 = 100% random actions

    for i in tqdm(range(episodes)):
        # Reset environment at teh beginning of episode
        state = env.reset()[0]
        terminated = False

        # Robot keeps going until it finds the target
        while(not terminated):

            # Select action based on epsilon-greedy
            if is_training and random.random() < epsilon:
                # select random action
                action = env.action_space.sample()
            else:
                # Convert state of [1,2,3,4] to (1,2,3,4), use this to index into the 4th dimension of the 5D array.
                q_state_idx = tuple(state)

                # select best action
                action = np.argmax(q[q_state_idx])

            # Perform action
            new_state,reward,terminated,_,_ = env.step(action)

            # Convert state of [1,2,3,4] and action of [1] into (1,2,3,4,1), use this to index into the 5th dimension of the 5D array.
            q_state_action_idx = tuple(state) + (action,)

            # Convert new_state of [1,2,3,4] into (1,2,3,4), use this to index into the 4th dimension of the 5D array.
            q_new_state_idx = tuple(new_state)

            if is_training:
                # Update Q-Table
                q[q_state_action_idx] = q[q_state_action_idx] + learning_rate_a * (
                        reward + discount_factor_g * np.max(q[q_new_state_idx]) - q[q_state_action_idx]
                )

            # Update current state
            state = new_state

        # Decrease epsilon
        epsilon = max(epsilon - 1/episodes, 0)


    episode_rewards = np.array(env.return_queue)
    episode_lengths = np.array(env.length_queue) # Reverse the list

    env.close()



    # Graph steps
    if is_training:
        sum_steps = np.zeros(episodes)
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_steps[t] = np.mean(episode_lengths[max(0, t-100):(t+1)]) # Average steps per 100 episodes
            sum_rewards[t] = np.mean(episode_rewards[max(0, t-100):(t+1)]) # Average steps per 100 episodes


        plt.figure(figsize=(15, 5))

        # Plot Episode Rewards
        plt.subplot(1, 2, 1)
        plt.plot(sum_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')

        # Plot Episode Lengths
        plt.subplot(1, 2, 2)
        plt.plot(sum_steps)
        plt.title('Episode Lengths')
        plt.xlabel('Episodes')
        plt.ylabel('Length')

        plt.tight_layout()
        plt.savefig('v0_warehouse_solution.png')


        # Save Q Table
        f = open("v0_warehouse_solution.pkl","wb")
        pickle.dump(q, f)
        f.close()


def make_env():
    # env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None)
    # env = gym.make("gymnasium_env/GridWorld-v0", render_mode="rgb_array")
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None)
    # env = RGBImgObsWrapper(env)
    # env = ImgObsWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    return env



def train_sb3():

    # maze = MazeGenerator(15)
    # maze.create_maze()
    # obstacles = maze.get_obstacle_coordinates()
    # obstacles = [
    #     (0, 3), (0, 6), (0, 12),
    #     (1, 2), (1, 5), (1, 8), (1, 10),  # Row 1 obstacles
    #     (2, 4), (2, 7), (2, 11), (2, 13),  # Row 2 obstacles
    #     (3, 3), (3, 6), (3, 9), (3, 12),  # Row 3 obstacles
    #     (4, 1), (4, 5), (4, 8), (4, 14),  # Row 4 obstacles
    #     (5, 2), (5, 7), (5, 13),          # Row 5 obstacles
    #     (6, 3), (6, 10), (6, 12),          # Row 6 obstacles
    #     (7, 1), (7, 6), (7, 11), (7, 13),  # Row 7 obstacles
    #     (8, 0), (8, 4), (8, 7), (8, 9),    # Row 8 obstacles
    #     (9, 2), (9, 5), (9, 8), (9, 14),   # Row 9 obstacles
    #     (10, 0), (10, 4), (10, 5), (10, 6), (10, 12), # Row 10 obstacles
    #     (11, 1), (11, 5), (11, 9), (11, 13), # Row 11 obstacles
    #     (12, 3), (12, 8), (12, 10), (12, 14), # Row 12 obstacles
    #     (13, 4), (13, 7), (13, 11),        # Row 13 obstacles
    #     (14, 2), (14, 5), (14, 9), (14, 12)  # Row 14 obstacles
    # ]

    # if not os.path.exists("obstacles.json"):
    #     with open("obstacles.json", "w") as file:
    #         json.dump(obstacles, file)

    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    #env = gym.make('gymnasium_env/GridWorld-v0', render_mode=None)
    # env = gym.make('gymnasium_env/GridWorld-v0', render_mode=None)
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    #env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = VecMonitor(SubprocVecEnv([make_env for _ in range(8)]))
    env = make_vec_env(make_env, n_envs=4, vec_env_cls=SubprocVecEnv)
    # env = VecTransposeImage(env)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)



    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    # env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None)
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    # model = DQN('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir, buffer_size=1500000)
    # policy_kwargs = dict(activation_fn=torch.nn.Mish,
    #                      net_arch=dict(pi=[8, 8], vf=[8, 8]))


    TIMESTEPS = 100000

    latest_model_path, iters = get_latest_model_and_iters(model_dir, TIMESTEPS)

    if latest_model_path:  # If a pre-trained model exists
        print(f"Loading existing model: {latest_model_path}")
        model = MaskablePPO.load(latest_model_path, env=env, tensorboard_log=log_dir, device='cpu')
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, device='cpu', tensorboard_log=log_dir,
                             n_epochs=8,
                             learning_rate=0.0001,
                             ent_coef=0.01,
                             n_steps=5120,
                             # clip_range=0.1,
                             # vf_coef=0.4,
                             # batch_size=2048,
                             # gae_lambda=0.95,
                             # gamma=0.99,
                             seed=7,
                             # policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))  # New way
                            # policy_kwargs=dict(net_arch=dict(pi=[256, 128, 128], vf=[256, 128, 128]))
                            # policy_kwargs = dict(
                            #     features_extractor_class=CustomCNNFeatureExtractor,
                            #     features_extractor_kwargs=dict(features_dim=512),
                            # )
                            )
        iters = 0

    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    # model_name = model.__class__.__name__

    if latest_model_path:
        file_name = os.path.basename(latest_model_path)
        model_name = file_name.rsplit('_', 1)[0]
    else:
        model_name = model.__class__.__name__
        unique_id = uuid.uuid4().hex  # Generates a unique identifier
        model_name = f"{model_name}_{unique_id}"

    while True:
        # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        # model.save(f"{model_dir}/{model_name}_{TIMESTEPS*iters}") # Save a trained model every TIMESTEPS

        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            callback=SaveOnTimestepCallback(model, model_dir, save_interval=TIMESTEPS, model_name=model_name),
            tb_log_name=model_name
        )

        if (iter == 0):
            latest_model_path, iters = get_latest_model_and_iters(model_dir, TIMESTEPS)
            file_name = os.path.basename(latest_model_path)
            model_name = file_name.rsplit('_', 1)[0]

        iters += 1



def test_sb3(render=True):

    env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")
    #env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False, map_name="8x8")

    # Load model
    model = MaskablePPO.load('models/MaskablePPO_837288', env=env)

    # Run a test
    for _ in range(20):
        obs, info = env.reset()
        # terminated = False
        while True:
            action_masks = get_action_masks(env)
            # print(action_masks)
            action, _ = model.predict(observation=obs, deterministic=True, action_masks=action_masks) # Turn on deterministic, so predict always returns the same behavior
            #debug_print(action)
            # print(action)
            obs, _, terminated, _, _ = env.step(action.item())


            if terminated:
                break

    env.close()

# ------------- q learning -------------
# run_q(20000, is_training=True, render=False)
# run_q(10, is_training=False, render=True)

# ------------- sb3 -------------
if __name__ == '__main__':
    train_sb3()

# test_sb3()