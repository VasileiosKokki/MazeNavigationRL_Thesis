import argparse
import os
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis

from stable_baselines3.common.env_util import make_vec_env
from agent import GridWorldEnv
import utils

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv


def save_best_hyperparams(study, folder):
    config_dir = os.path.join("configs", folder)
    model_config_path = utils.get_config_path(config_dir, "model_config.py")
    config = utils.load_config_from_py(model_config_path)
    policy_name = config.policy_name

    best = study.best_trial

    batch_size_pow = best.params.get("batch_size_pow")
    n_steps_pow = best.params.get("n_steps_pow")
    one_minus_gamma = best.params.get("one_minus_gamma")
    one_minus_gae_lambda = best.params.get("one_minus_gae_lambda")
    learning_rate = best.params.get("learning_rate")  # or 'lr' if named like that in trial
    ent_coef = best.params.get("ent_coef")
    clip_range = best.params.get("clip_range")
    n_epochs = best.params.get("n_epochs")
    max_grad_norm = best.params.get("max_grad_norm")

    # Derived hyperparameters
    batch_size = 2 ** batch_size_pow
    n_steps = 2 ** n_steps_pow
    gamma = 1.0 - one_minus_gamma
    gae_lambda = 1.0 - one_minus_gae_lambda

    best_hyperparams = dict(
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        max_grad_norm=max_grad_norm,
    )

    if "policy_kwargs" in config.hyperparams:
        best_hyperparams["policy_kwargs"] = config.hyperparams["policy_kwargs"]

    utils.save_model_config(policy_name, best_hyperparams, config_dir)

def save_optuna_plots(study, folder):
    optuna_log_dir = os.path.join("optuna_logs", folder)
    os.makedirs(optuna_log_dir, exist_ok=True)

    # 1. Optimization History
    ax1 = vis.plot_optimization_history(study)
    fig1 = ax1.figure
    fig1.savefig(os.path.join(optuna_log_dir, "optimization_history.png"))

    # 2. Hyperparameter Importance
    ax2 = vis.plot_param_importances(study)
    fig2 = ax2[0].get_figure() if isinstance(ax2, np.ndarray) else ax2.figure # type: ignore[attr-defined]
    fig2.savefig(os.path.join(optuna_log_dir, "param_importance.png"))

    # 3. Slice Plot (shows influence of each hyperparam on objective)
    ax3 = vis.plot_slice(study)
    fig3 = ax3[0].figure if isinstance(ax3, np.ndarray) else ax3.figure # type: ignore[attr-defined]
    fig3.savefig(os.path.join(optuna_log_dir, "param_slices.png"))

    plt.close('all')  # Free up memory
    print(f"✅ Saved plots to '{optuna_log_dir}'")


def evaluate(model, env, n_episodes=3):
    total_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = np.array([False])
        ep_reward = 0
        while not done.any():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0]  # reward is an array from VecEnv
        total_rewards.append(ep_reward)
    return np.mean(total_rewards)



def objective(trial):

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None,
                        help="name of the folder")
    args = parser.parse_args()
    config_dir = os.path.join("configs", args.folder)
    env_config_path = utils.get_config_path(config_dir, "env_config.py")
    print(f"Loading existing env config: {env_config_path}")
    config = utils.load_config_from_py(env_config_path)
    env_kwargs = config.env_kwargs
    print(env_kwargs)
    use_frame_stacking = config.use_frame_stacking


    env = make_vec_env(utils.make_env(**env_kwargs), n_envs=8, seed=42, vec_env_cls=SubprocVecEnv)
    if use_frame_stacking:
        env = VecFrameStack(env, n_stack=4, channels_order='last')

    # Hyperparameter search spaces
    batch_size_pow = trial.suggest_int("batch_size_pow", 6, 8)  # 2^6=64 to 2^8=256 (more stable batch sizes)
    n_steps_pow = trial.suggest_int("n_steps_pow", 7, 11)       # 2^7=128 to 2^10=1024 (typical update steps)
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0005, 0.01, log=True)  # gamma ~0.99 to 0.9995
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.001, 0.05, log=True)  # gae_lambda ~0.95 to 0.999
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)  # smaller LR range for stability
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.01, log=True)  # lower entropy coef range
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])  # skip 0.4 as it’s quite large
    n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10])  # a bit less aggressive
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 1.0)  # tighter clipping range

    # Derived hyperparameters
    batch_size = 2 ** batch_size_pow
    n_steps = 2 ** n_steps_pow
    gamma = 1.0 - one_minus_gamma
    gae_lambda = 1.0 - one_minus_gae_lambda

    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("n_steps", n_steps)
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("gae_lambda", gae_lambda)

    model_config_path = utils.get_config_path(config_dir, "model_config.py")
    print(f"Loading existing model config: {model_config_path}")
    config = utils.load_config_from_py(model_config_path)
    policy_name = config.policy_name

    model_kwargs = dict(
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        max_grad_norm=max_grad_norm,
    )

    if "policy_kwargs" in config.hyperparams:
        model_kwargs["policy_kwargs"] = config.hyperparams["policy_kwargs"]

    model = PPO(policy_name, env, verbose=0, device='cuda', **model_kwargs)

    eval_interval = 200_000  # fewer evals, faster tuning
    total_timesteps = 400_000

    mean_reward = 0
    for step in range(eval_interval, total_timesteps + 1, eval_interval):
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        mean_reward = evaluate(model, env)

        trial.report(mean_reward, step)
        if trial.should_prune():
            env.close()
            raise optuna.exceptions.TrialPruned()

    env.close()
    return mean_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None,
                        help="name of the folder")
    args = parser.parse_args()
    folder = args.folder

    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=6, n_jobs=2, show_progress_bar=True)

    print("Best trial:")
    print(study.best_trial)
    save_optuna_plots(study, folder)
    save_best_hyperparams(study, folder)




