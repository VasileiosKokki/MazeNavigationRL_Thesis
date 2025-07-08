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
from torchview import draw_graph

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
from python import callbacks

console = Console()


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)

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



def train_sb3():

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None,
                        help="name of the folder")
    args = parser.parse_args()

    utils.seed(42)

    # Where to store trained model and logs
    model_dir = os.path.join("models", args.folder)
    log_dir = os.path.join("logs")
    config_dir = os.path.join("configs", args.folder)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    latest_model_path = utils.get_latest_model_path(model_dir)
    model_config_path = utils.get_config_path(config_dir, "model_config.py")
    env_config_path = utils.get_config_path(config_dir, "env_config.py")

    policy_name = "MlpPolicy"



    if env_config_path:
        print(f"Loading existing env config: {env_config_path}")
        config = utils.load_config_from_py(env_config_path)
        env_kwargs = config.env_kwargs
        use_frame_stacking = config.use_frame_stacking
    else:
        env_kwargs = {
            "size": 10,
            "num_obstacles": 15,
            "num_patterns": 10,
            "moving_target": False,
            "dense_rewards": True,
            "policy": policy_name
        }

        use_frame_stacking = False

        utils.save_env_config(env_kwargs, use_frame_stacking, config_dir)

    env = make_vec_env(utils.make_env(**env_kwargs), n_envs=8, seed=42, vec_env_cls=SubprocVecEnv)
    if use_frame_stacking:
        env = VecFrameStack(env, n_stack=4, channels_order='last')

    TIMESTEPS = 10000

    if latest_model_path:  # If a pre-trained model exists
        print(f"Loading existing model: {latest_model_path}")
        model = PPO.load(latest_model_path, env=env, tensorboard_log=log_dir, device=utils.get_device(env_kwargs['policy']))
    else:
        if model_config_path:  # only if config exists but not model, in case we modify the existing config and we delete the models
            print(f"Loading existing model config: {model_config_path}")
            config = utils.load_config_from_py(model_config_path)
            policy_name = config.policy_name
            hyperparams = config.hyperparams
            print(hyperparams)
        else:  # default fallback, no model and no config
            hyperparams = dict(
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=0.0003,
                ent_coef=0.00,
                clip_range=0.2,
                n_epochs=10,
                max_grad_norm=0.5,
                policy_kwargs=dict(
                    # features_extractor_class=CustomCNNFeatureExtractor,
                    # features_extractor_kwargs=dict(features_dim=512),
                    net_arch=dict(pi=[64], vf=[64]),
                    activation_fn=nn.Tanh
                ),
            )

            utils.save_model_config(policy_name, hyperparams, config_dir)

        model = PPO(policy_name, env, verbose=1, device=utils.get_device(policy_name), tensorboard_log=log_dir, seed=42, **hyperparams)

    print(model.policy)

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
        save_cb = callbacks.SaveOnTimestepCallback(model, model_dir, save_interval=TIMESTEPS, model_name=model_name)
        mean_goal_cb = callbacks.MeanGoalAchievedCallback()
        max_length_cb = callbacks.MaxEpisodeLengthCallback()
        max_wrong_steps_cb = callbacks.MaxWrongStepsCallback()



        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            callback=[save_cb, mean_goal_cb, max_length_cb, max_wrong_steps_cb],
            tb_log_name=model_name
        )


def test_sb3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None,
                        help="name of the folder")
    args = parser.parse_args()

    utils.seed(42)

    model_dir = os.path.join("models", args.folder)
    config_dir = os.path.join("configs", args.folder)
    env_config_path = utils.get_config_path(config_dir, "env_config.py")

    latest_model_path = utils.get_latest_model_path(model_dir)
    print(latest_model_path)

    config = utils.load_config_from_py(env_config_path)
    env_kwargs = config.env_kwargs
    use_frame_stacking = config.use_frame_stacking

    if not use_frame_stacking:
        env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human", **env_kwargs)

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
    else:
        env = make_vec_env(utils.make_env(render_mode="human", **env_kwargs), n_envs=1, seed=42, vec_env_cls=DummyVecEnv)
        env = VecFrameStack(env, n_stack=4, channels_order='last')

        model = PPO.load(f'{latest_model_path}', env=env)

        print(model.policy)
        # Run a test
        for _ in range(20):
            obs = env.reset()

            # print(obs)
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



# ------------- sb3 -------------
if __name__ == '__main__':
    # train_sb3()
    test_sb3()
