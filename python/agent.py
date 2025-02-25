import argparse
import pickle
import random
import re
import shutil
import sys
import json
import time
import uuid
from rich.console import Console

from gymnasium.spaces import Dict

import utils

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
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv, VecTransposeImage
from sb3_contrib import QRDQN
from sb3_contrib import RecurrentPPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from torch import nn, Tensor
from stable_baselines3.common.vec_env import VecFrameStack

import gymnasium_env
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import TimeLimit

# from agent import GridWorldAgent
from tqdm import tqdm

from gymnasium_env.envs import GridWorldEnv

console = Console()


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
            # Delete previous checkpoint
            shutil.rmtree(self.save_path)
            os.makedirs(self.save_path)

            self.model.save(checkpoint_path)

        return True





class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)


        # assert isinstance(observation_space, Dict), "Observation space must be a Dict space"
        # n_input_channels = observation_space.spaces["image"].shape[0]  # Get the number of image channels
        n_input_channels = observation_space.shape[0]  # Get the number of image channels
        print(n_input_channels)

        self.image_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )


    def forward(self, observations):
        x = self.image_conv(observations)
        return x




def get_latest_model_and_iters(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if not model_files:
        return None, 0

    # Sort by modification time, newest first
    model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model = model_files[0]
    path = os.path.join(model_dir, latest_model)

    return path


# def make_env(render_mode=None):
#     # env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None)
#     # env = gym.make("gymnasium_env/GridWorld-v0", render_mode="rgb_array")
#     env = gym.make("gymnasium_env/GridWorld-v0", render_mode=render_mode)
#     # env = RGBImgObsWrapper(env)
#     # env = ImgObsWrapper(env)
#     # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
#     return env

def make_env(render_mode=None):
    def _make_env():
        # Ensure the environment is created with the correct render_mode
        env = gym.make("gymnasium_env/GridWorld-v0", render_mode=render_mode)
        return env
    return _make_env  # Return the function



def train_sb3():

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None,
                        help="name of the folder")
    args = parser.parse_args()

    # load the obstacle patterns from the file
    file_path = 'obstacle_patterns.json'
    if not os.path.exists(file_path):
        GridWorldEnv().generate_pattern()

    # Where to store trained model and logs
    model_dir = os.path.join("models", args.folder)
    log_dir = os.path.join("logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # env = gym.make('gymnasium_env/GridWorld-v0', render_mode=None)
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = VecMonitor(SubprocVecEnv([make_env for _ in range(8)]))
    env = make_vec_env(make_env(), n_envs=16, vec_env_cls=SubprocVecEnv)
    # env = VecFrameStack(env, n_stack=8, channels_order='last')
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)


    TIMESTEPS = 10000

    latest_model_path = get_latest_model_and_iters(model_dir)

    if latest_model_path:  # If a pre-trained model exists
        print(f"Loading existing model: {latest_model_path}")
        model = PPO.load(latest_model_path, env=env, tensorboard_log=log_dir, device='cpu')
    else:
        model = PPO("CnnPolicy", env, verbose=1, device='cpu', tensorboard_log=log_dir,
                             n_epochs=10,
                             learning_rate=0.0003,
                             ent_coef=0.01,
                             clip_range=0.2,
                             # vf_coef=0.4,
                             batch_size=64,
                             gae_lambda=0.95,
                             gamma=0.99,
                             seed=7,
                             # policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))  # New way
                             # policy_kwargs=dict(net_arch=dict(pi=[256, 128, 128], vf=[256, 128, 128]))
                             policy_kwargs=dict(
                                 features_extractor_class=CustomCNNFeatureExtractor,
                                 features_extractor_kwargs=dict(features_dim=512),
                                 net_arch=dict(pi=[64], vf=[64]),
                                 # lstm_hidden_size=128
                             ),
                            )

    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.

    if latest_model_path:
        file_name = os.path.basename(latest_model_path)
        model_name = file_name.rsplit('_', 1)[0]
    else:
        model_name = model.__class__.__name__
        model_name = f"{model_name}_{args.folder}"

    while True:
        # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        # model.save(f"{model_dir}/{model_name}_{TIMESTEPS*iters}") # Save a trained model every TIMESTEPS

        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            callback=SaveOnTimestepCallback(model, model_dir, save_interval=TIMESTEPS, model_name=model_name),
            tb_log_name=model_name
        )


def frame_stack_test_sb3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None,
                        help="name of the folder")
    args = parser.parse_args()

    model_dir = os.path.join("models", args.folder)

    file_path = 'obstacle_patterns.json'
    if not os.path.exists(file_path):
        GridWorldEnv().generate_pattern()

    env = make_vec_env(make_env(render_mode="human"), n_envs=1, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=8, channels_order='last')

    latest_model_path = get_latest_model_and_iters(model_dir)
    print(latest_model_path)
    model = PPO.load(f'{latest_model_path}', env=env)

    # Run a test
    for _ in range(20):
        obs = env.reset()

        # print(obs)
        terminated = False
        while True:
            action, _ = model.predict(
                observation=obs,
                deterministic=True,
            )
            # obs, reward, terminated, truncated = env.step(np.array([action for _ in range(1)]))
            obs, reward, terminated, truncated = env.step(action)

            # unwrapped_env = env.envs[0].unwrapped
            # unstacked_obs = unwrapped_env._get_obs()
            # unstacked_obs = np.squeeze(unstacked_obs, axis=-1)
            # print(unstacked_obs)
            print(reward)

    env.close()


def test_sb3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None,
                        help="name of the folder")
    args = parser.parse_args()

    model_dir = os.path.join("models", args.folder)

    file_path = 'obstacle_patterns.json'
    if not os.path.exists(file_path):
        GridWorldEnv().generate_pattern()


    env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")

    latest_model_path = get_latest_model_and_iters(model_dir)
    print(latest_model_path)
    # Load model
    model = PPO.load(f'{latest_model_path}', env=env)

    # Run a test
    for _ in range(20):
        obs, info = env.reset()

        while True:
            # action_masks = get_action_masks(env)
            # print(action_masks)
            action, _ = model.predict(
                observation=obs,
                deterministic=True,
                # action_masks=action_masks
            )
            obs, reward, terminated, truncated, info = env.step(action.item())

            print(reward)
            if terminated or truncated:
                break

    env.close()


# ------------- sb3 -------------
# if __name__ == '__main__':
#     train_sb3()

if __name__ == '__main__':
    test_sb3()
    # frame_stack_test_sb3()