from __future__ import annotations

import glob
import os
import re
import time
from array import array


import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy
from pettingzoo.utils import parallel_to_aec, aec_to_parallel
from pettingzoo.test import api_test, parallel_api_test

from pettingzoo.sisl import waterworld_v4

from petting_env.envs import CustomActionMaskedEnvironment

def get_latest_model_and_iters(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if not model_files:
        return None

    # Sort by modification time, newest first
    model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model = model_files[0]

    return os.path.join(model_dir, latest_model)

def train_butterfly_supersuit(
        env_fn, steps: int = 10_000, seed: int | None = 0, render_mode = None
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    # env = env_fn.parallel_env()
    env = env_fn(render_mode=render_mode)
    env = aec_to_parallel(env)
    parallel_api_test(env, num_cycles=1000)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

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
            MlpPolicy,
            env,
            verbose=3,
            learning_rate=1e-3,
            batch_size=256,
            tensorboard_log=log_dir,
            device='cpu'
        )

    while True:
        model.learn(total_timesteps=steps, reset_num_timesteps=False)

        save_path = os.path.join(model_dir, f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
        model.save(save_path)

        # print("Model has been saved.")
        #
        # print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    # env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None):
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode)
    # env = parallel_to_aec(env)
    # api_test(env, num_cycles=1000)
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
        time.sleep(1)
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

            print(f"\nAgent: {agent}, \nObservation: \n{obs}, \nReward: {reward}, \nAction: {act}")
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    env_fn = CustomActionMaskedEnvironment
    # env_fn = waterworld_v4

    # Train a model (takes ~3 minutes on GPU)
    # train_butterfly_supersuit(env_fn, seed=0)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    # eval(env_fn, num_games=10)

    # Watch 2 games
    eval(env_fn, num_games=10, render_mode='human')