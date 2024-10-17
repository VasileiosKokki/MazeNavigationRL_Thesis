import pickle
import sys
import json

import numpy as np
import pydevd_pycharm

import gymnasium as gym
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

def train_agent(episodes):


    learning_rate = 0.01
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (episodes / 2)  # reduce the exploration over time
    final_epsilon = 0

    #env = gym.make('gymnasium_env/GridWorld-v0', render_mode=None)
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), render_mode=None, is_slippery=False)
    #env = gym.wrappers.TimeLimit(env, max_episode_steps=10000)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # agent = GridWorldAgent(
    #     env,
    #     learning_rate=learning_rate,
    #     initial_epsilon=start_epsilon,
    #     epsilon_decay=epsilon_decay,
    #     final_epsilon=final_epsilon,
    # )
    agent = None

    episode_error = [0] * episodes
    is_training = True
    rewards_per_episode = np.zeros(episodes)

    for episode in tqdm(range(episodes)):
        if episode == episodes - 3:  # Check if it's the last episode
            episode_rewards = env.return_queue  # Rewards per episode
            episode_lengths = env.length_queue    # Lengths per episode
            #env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")
            random_map = generate_random_map(size=8)
            env = gym.make('FrozenLake-v1', desc=random_map, render_mode="human", is_slippery=False)
            #env = gym.wrappers.TimeLimit(env, max_episode_steps=10000)
            is_training = False


        obs, info = env.reset()
        #debug_print(obs)
        done = False


        # play one episode
        while not done:
            action = agent.get_action(obs, is_training)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent if training
            if is_training:
                agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
        if reward == 1:
            rewards_per_episode[episode] = 1
        episode_error[episode] = sum(agent.training_error)
        #agent.training_error = []




    #debug_print(env.return_queue)
    env.close()
    # Plotting the statistics
    plt.figure(figsize=(15, 5))

    # Plot Episode Rewards
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')

    # Plot Episode Lengths
    plt.subplot(1, 3, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episodes')
    plt.ylabel('Length')

    # Plotting Training Error (If applicable)
    # For demonstration, let's plot random data for training error
    plt.subplot(1, 3, 3)
    plt.plot(agent.training_error)  # Replace with actual training error
    plt.title('Training Error')
    plt.xlabel('Episodes')
    plt.ylabel('Error')

    plt.tight_layout()
    plt.show()

def train_sb3():
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    #env = gym.make('gymnasium_env/GridWorld-v0', render_mode=None)
    env = gym.make('gymnasium_env/GridWorld-v0', render_mode=None)
    #env = gym.wrappers.RecordEpisodeStatistics(env)

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = A2C('MultiInputPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/a2c_{TIMESTEPS*iters}") # Save a trained model every TIMESTEPS



def test_sb3(render=True):

    env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")
    #env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False, map_name="8x8")

    # Load model
    model = A2C.load('models/a2c_10000', env=env)

    # Run a test
    for _ in range(3):
        obs, info = env.reset()
        terminated = False
        while True:
            action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
            #debug_print(action)
            obs, _, terminated, _, _ = env.step(action.item())


            if terminated:
                break

    env.close()

# q learning
#train_agent(1000)

# sb3
#train_sb3()
test_sb3()