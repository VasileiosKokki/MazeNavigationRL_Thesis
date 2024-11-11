import pickle
import random
import sys
import json
import time

import numpy as np
import pydevd_pycharm

import gymnasium as gym
import torch
from matplotlib import pyplot as plt
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import A2C, PPO, DQN
import os
import tensorflow as tf

from gym.envs.toy_text.frozen_lake import generate_random_map
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from sb3_contrib import QRDQN
from sb3_contrib import RecurrentPPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.envs import InvalidActionEnvDiscrete


import gymnasium_env
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import TimeLimit

# from agent import GridWorldAgent
from tqdm import tqdm

from gymnasium_env.envs import GridWorldEnv


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
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    return env

def train_sb3():
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
    env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    # env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None)
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    # model = DQN('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir, buffer_size=1500000)
    policy_kwargs = dict(activation_fn=torch.nn.Mish,
                         net_arch=dict(pi=[8, 8], vf=[8, 8]))


    model = MaskablePPO("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log=log_dir,
                         learning_rate=0.0001,
                         # clip_range=0.1,
                         # vf_coef=0.5,
                         # n_steps=128,
                         batch_size=128,
                         # n_epochs=4,
                         ent_coef=0.01,
                         # gae_lambda=0.95,
                         # gamma=0.4,
                         # seed=7
                         # policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))  # New way
                         policy_kwargs=policy_kwargs
                        )

    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    model_name = model.__class__.__name__
    TIMESTEPS = 100000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{model_name}_{TIMESTEPS*iters}") # Save a trained model every TIMESTEPS



def test_sb3(render=True):

    env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")
    #env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False, map_name="8x8")

    # Load model
    model = MaskablePPO.load('models/MaskablePPO_3000000', env=env)

    # Run a test
    for _ in range(20):
        obs, info = env.reset()
        terminated = False
        while True:
            action_masks = get_action_masks(env)
            print(action_masks)
            action, _ = model.predict(observation=obs, deterministic=True, action_masks=action_masks) # Turn on deterministic, so predict always returns the same behavior
            #debug_print(action)
            print(action)
            obs, _, terminated, _, _ = env.step(action.item())


            if terminated:
                break

    env.close()

# q learning
# run_q(2000, is_training=True, render=False)
# run_q(10, is_training=False, render=True)

# sb3
# if __name__ == '__main__':
#     train_sb3()

test_sb3()