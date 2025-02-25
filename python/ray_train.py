import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
import gymnasium as gym
from gym_world.gymnasium_env.envs.grid_world import GridWorldEnv


# Initialize Ray
ray.init(ignore_reinit_error=True)

# Define your custom environment registration function
def make_env(env_config=None):
    # env = gym.make("gymnasium_env/GridWorld-v0")
    env = GridWorldEnv()
    return env

# Register environment in RLlib
register_env("GridWorld-v0", lambda config: make_env(config))

# Load latest checkpoint if available
def get_latest_checkpoint(logdir):
    checkpoints = [f for f in os.listdir(logdir) if f.startswith("checkpoint")]
    if not checkpoints:
        return None
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1]
    return os.path.join(logdir, latest_checkpoint, "checkpoint-1")

# Training configuration
config = (
    PPOConfig()
    .environment(env="GridWorld-v0", env_config={})
    .framework("torch")  # Use PyTorch
    .api_stack(enable_rl_module_and_learner=True)  # Disable the new RLModule stack
    .training(
        gamma=0.99,
        lambda_=0.95,
        lr=0.0003,
        num_epochs=10,
        train_batch_size=64,
        minibatch_size=64,
        model={
            "conv_filters": [  # Custom CNN architecture
                [16, [2, 2], 1],  # 16 filters, 3x3 kernel, stride 1
                [32, [2, 2], 1],  # 32 filters, 3x3 kernel, stride 1
                [64, [2, 2], 1],  # 64 filters, 3x3 kernel, stride 1
            ],
            "conv_activation": "relu",  # Activation function
            "use_lstm": False,  # Disable LSTM since error suggests lstm=False
            "fcnet_hiddens": [64],  # Fully connected layers
            "fcnet_activation": "relu",
            "vf_hiddens": [64],  # Fully connected layers for the critic
            "vf_activation": "relu",
        },
    ).env_runners(num_envs_per_env_runner=2, num_env_runners=2)
)

# Create PPO trainer
trainer = config.build()

# Check for existing checkpoint
checkpoint_dir = "rllib_models"
os.makedirs(checkpoint_dir, exist_ok=True)
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f"Loading checkpoint: {latest_checkpoint}")
    trainer.restore(latest_checkpoint)

# Training Loop
TIMESTEPS = 100000
iters = 0

while True:
    result = trainer.train()
    episode_len_mean = result['env_runners']['episode_len_mean']
    episode_len_max = result['env_runners']['episode_len_max']
    episode_return_mean = result['env_runners']['episode_return_mean']
    entropy = result['learners']['default_policy']['entropy']
    print(f"Iteration: {iters}, Reward: {episode_return_mean}, Length: {episode_len_mean}, MaxLength: {episode_len_max}, Entropy: {entropy}")

    if iters % 100 == 0:  # Save model every 100 iterations
        checkpoint_dir = 'C:\\Users\\User\\Desktop\\Personal Project\\tankio-master\\python'
        os.makedirs(checkpoint_dir, exist_ok=True)  # Create the directory if it doesn't exist
        checkpoint_path = trainer.save(checkpoint_dir)
        print(f"Checkpoint saved at {checkpoint_path}")

    iters += 1
