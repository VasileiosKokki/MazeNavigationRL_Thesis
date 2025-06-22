import importlib.util
import inspect
import os
import random
import shutil
import textwrap

import gymnasium as gym
import numpy
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tensorboardX import SummaryWriter


def get_latest_model_path(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if not model_files:
        return None

    # Sort by modification time, newest first
    model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model = model_files[0]
    path = os.path.join(model_dir, latest_model)

    return path


def get_config_path(model_dir, config_file):
    full_path = os.path.join(model_dir, config_file)

    if os.path.isfile(full_path):
        return full_path
    return None


def save_model_config(policy_name, hyperparams, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "model_config.py")

    def convert(val):
        if hasattr(val, "__module__") and hasattr(val, "__name__"):
            # Just use class name (no module prefix)
            if val.__module__.startswith("torch.nn"):
                 return f"nn.{val.__name__}"
            return val.__name__
        elif isinstance(val, dict):
            inner = ",\n".join(f"{k}={convert(v)}" for k, v in val.items())
            return f"dict(\n{textwrap.indent(inner, '    ')}\n)"
        elif isinstance(val, list):
            return "[" + ", ".join(convert(v) for v in val) + "]"
        elif isinstance(val, str):
            return f'"{val}"'
        else:
            return repr(val)

    with open(file_path, "w") as f:
        f.write("from torch import nn\n")
        f.write(f"from python.agent import CustomCNNFeatureExtractor\n\n")

        f.write(f'policy_name = "{policy_name}"\n\n')
        f.write("hyperparams = " + convert(hyperparams) + "\n")

def save_env_config(params, use_frame_stacking, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "env_config.py")

    with open(file_path, "w") as f:
        f.write("env_kwargs = {\n")
        for key, value in params.items():
            if isinstance(value, str):
                f.write(f'    "{key}": "{value}",\n')
            else:
                f.write(f'    "{key}": {value},\n')
        f.write("}\n")

        f.write(f"use_frame_stacking = {use_frame_stacking}\n")


def load_config_from_py(path):
    module_name = os.path.splitext(os.path.basename(path))[0]  # e.g., "model_config"
    spec = importlib.util.spec_from_file_location(module_name, path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def make_env(render_mode=None, **env_kwargs):
    def _make_env():
        env = gym.make("gymnasium_env/GridWorld-v0", render_mode=render_mode, **env_kwargs)
        return env
    return _make_env  # Return the function



def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



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



class MeanGoalAchievedCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.final_rewards = []

    def _on_step(self) -> bool:
        done = self.locals.get("dones")  # vector of done flags or bool
        rewards = self.locals.get("rewards")

        if done is not None and rewards is not None:
            # handle vectorized env case
            if isinstance(done, (list, tuple)) or hasattr(done, "__len__"):
                for d, r in zip(done, rewards):
                    if d:  # episode done
                        clipped_reward = 1 if r > 0 else 0
                        self.final_rewards.append(clipped_reward)
            else:
                if done:
                    clipped_reward = 1 if rewards > 0 else 0
                    self.final_rewards.append(clipped_reward)
        return True

    def _on_rollout_end(self) -> None:
        # Log mean final reward to TensorBoard at rollout end
        if self.final_rewards:
            mean_final_reward = np.mean(self.final_rewards)
            # self.logger is the Stable Baselines3 logger for TensorBoard
            self.logger.record("ep_goal_mean", mean_final_reward)
            self.final_rewards.clear()  # reset for next rollout