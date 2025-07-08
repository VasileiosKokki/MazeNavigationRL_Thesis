import os
import shutil

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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
            for d, r in zip(done, rewards):
                if d:  # episode done
                    clipped_reward = 1 if r > 0 else 0
                    self.final_rewards.append(clipped_reward)

        return True

    def _on_rollout_end(self) -> None:
        # Log mean final reward to TensorBoard at rollout end
        if self.final_rewards:
            mean_final_reward = np.mean(self.final_rewards)
            # self.logger is the Stable Baselines3 logger for TensorBoard
            self.logger.record("ep_goal_mean", mean_final_reward)
            self.final_rewards.clear()  # reset for next rollout


class MaxEpisodeLengthCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.max_length = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")

        for info in infos:
            length = info.get("episode", {}).get("l")
            if length is not None and length > self.max_length:
                self.max_length = length

        return True

    def _on_rollout_end(self) -> None:
        if self.max_length > 0:
            self.logger.record("ep_len_max", self.max_length)
            self.max_length = 0  # reset for the next rollout


class MaxWrongStepsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_wrong_steps = 0
        self.max_wrong_steps = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        for done, info in zip(dones, infos):
            if done and "wrong_steps" in info:
                if info["wrong_steps"] > self.max_wrong_steps:
                    self.max_wrong_steps = info["wrong_steps"]

        return True

    def _on_rollout_end(self) -> None:
        if self.max_wrong_steps > 0:
            self.logger.record("ep_wrongSteps_max", self.max_wrong_steps)
            self.max_wrong_steps = 0
