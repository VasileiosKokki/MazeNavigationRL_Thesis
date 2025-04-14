from __future__ import annotations

import glob
import os
import re
import time
from array import array

from gymnasium.utils import seeding
from pettingzoo.classic import tictactoe_v3
from magent2.environments import adversarial_pursuit_v4
from pettingzoo.butterfly import knights_archers_zombies_v10


import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy, CnnPolicy
from pettingzoo.utils import parallel_to_aec, aec_to_parallel
from pettingzoo.test import api_test, parallel_api_test

from pettingzoo.sisl import waterworld_v4
from torch import nn

from petting_env.envs import parallel_env
# from petting_env.envs import CustomActionMaskedEnvironment

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import BaseCallback
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_lengths = {}
        self.episode_rewards = {}
        self.env = None

    def _on_training_start(self) -> None:
        self.env = self.training_env

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

                # Initialize total reward and total length for agent1 and agent2
        total_reward_agent1 = 0.0
        total_length_agent1 = 0
        total_reward_agent2 = 0.0
        total_length_agent2 = 0

        # Iterate over each agent's info
        for agent_info in infos:
            agent_id = agent_info.get("agent_id")  # Get the actual agent_id
            reward = agent_info.get("reward")
            done = agent_info.get("done")

            # Track cumulative reward and length for each agent
            self.episode_rewards[agent_id] = self.episode_rewards.get(agent_id, 0.0) + (reward if reward is not None else 0.0)
            self.episode_lengths[agent_id] = self.episode_lengths.get(agent_id, 0) + 1

            if done:  # If the agent's episode is done (either terminated or truncated)
                # Log the episode reward and length
                # self.logger.record(f"{agent_id}/episode_reward", self.episode_rewards[agent_id])
                # self.logger.record(f"{agent_id}/episode_length", self.episode_lengths[agent_id])

                # Add rewards and lengths based on agent_id: one for agent1, the other for agent2
                if agent_id == "guard1":  # Handling for agent_1
                    total_reward_agent1 += self.episode_rewards[agent_id]
                    total_length_agent1 += self.episode_lengths[agent_id]
                elif agent_id == "guard2":  # Handling for agent_2
                    total_reward_agent2 += self.episode_rewards[agent_id]
                    total_length_agent2 += self.episode_lengths[agent_id]


        # Reset the counters for that agent
                self.episode_rewards[agent_id] = 0.0
                self.episode_lengths[agent_id] = 0

        # Compute the mean reward and mean length for each agent (across the 16 environments)
        mean_reward_agent1 = total_reward_agent1 / 12
        mean_length_agent1 = total_length_agent1 / 12

        mean_reward_agent2 = total_reward_agent2 / 12
        mean_length_agent2 = total_length_agent2 / 12

        # Log the mean reward and mean length for agent1 and agent2
        self.logger.record("agent1/mean_episode_reward", mean_reward_agent1)
        self.logger.record("agent1/mean_episode_length", mean_length_agent1)

        self.logger.record("agent2/mean_episode_reward", mean_reward_agent2)
        self.logger.record("agent2/mean_episode_length", mean_length_agent2)


        return True







def get_latest_model_and_iters(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if not model_files:
        return None

    # Sort by modification time, newest first
    model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model = model_files[0]

    return os.path.join(model_dir, latest_model)


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

def train_butterfly_supersuit(
        env_fn, steps: int = 10_000, seed: int | None = 0, render_mode = None
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    # env = env_fn.parallel_env()
    env = env_fn(render_mode=render_mode)
    # env = aec_to_parallel(env)
    parallel_api_test(env, num_cycles=1000)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    log_dir = "logs/"
    model_dir = "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    latest_policy = get_latest_model_and_iters(model_dir)
    if latest_policy:
        model = PPO.load(latest_policy, env=env, tensorboard_log=log_dir, device='cpu')
    else:
        # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
        model = PPO(
            # CnnPolicy,
            MlpPolicy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device='cpu',
            n_epochs=10,
            learning_rate=0.0003,
            ent_coef=0.01,
            clip_range=0.2,
            # vf_coef=0.4,
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            # seed=7,
            # policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))  # New way
            # policy_kwargs=dict(net_arch=dict(pi=[256, 128, 128], vf=[256, 128, 128]))
            # policy_kwargs=dict(
            #     features_extractor_class=CustomCNNFeatureExtractor,
            #     features_extractor_kwargs=dict(features_dim=512),
            #     net_arch=dict(pi=[64], vf=[64]),
            #     # lstm_hidden_size=128
            # ),
        )

    while True:
        model.learn(total_timesteps=steps,
                    # reset_num_timesteps=False,
                    # callback=TensorboardCallback()
                    )

        save_path = os.path.join(model_dir, f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
        model.save(save_path)

        # print("Model has been saved.")
        #
        # print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    # env.close()
from pettingzoo.utils.wrappers import BaseWrapper
class AutoDeadStepWrapper(BaseWrapper):
    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # If agent is done, ignore given action and pass None
            super().step(None)
        else:
            super().step(action)

def eval(env_fn, num_games: int = 100, render_mode: str | None = None):
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode)
    # parallel_api_test(env, num_cycles=1000)
    env = parallel_to_aec(env)
    # env = AutoDeadStepWrapper(env)
    # env.reset()
    api_test(env, num_cycles=1000)
    # env = env_fn.env(render_mode=render_mode)

    print(
        f"\nStarting evaluation on '{str(env.metadata['name'])}' (num_games={num_games}, render_mode={render_mode})"
    )

    # try:
    #     model_dir = "models"  # Specify the directory containing the model files
    #     latest_policy = max(
    #         glob.glob(os.path.join(model_dir, f"{env.metadata['name']}*.zip")),
    #         key=os.path.getctime,
    #     )
    # except ValueError:
    #     print("Policy not found.")
    #     exit(0)
    model_dir = "models"
    latest_policy = get_latest_model_and_iters(model_dir)
    if latest_policy:
        model = PPO.load(latest_policy, device='cpu')
    else:
        print("Policy not found.")
        exit(0)

    # print(env.possible_agents)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        # time.sleep(1)
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            # print(obs)

            if reward > 0:
                rewards[agent] += reward

            if termination or truncation:
                act = None
            else:
                act = model.predict(obs, deterministic=True)[0]
                act = act.item()

            # print(f"\nAgent: {agent}, \nObservation: \n{obs}, \nReward: {reward}, \nAction: {act}")
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    # env_fn = CustomActionMaskedEnvironment
    env_fn = parallel_env
    # env_fn = waterworld_v4

    # Train a model (takes ~3 minutes on GPU)
    train_butterfly_supersuit(env_fn, seed=0)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    # eval(env_fn, num_games=10)

    # Watch 2 games
    # eval(env_fn, num_games=10, render_mode='human')