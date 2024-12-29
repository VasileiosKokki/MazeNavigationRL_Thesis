import os

import numpy as np
import torch
from petting_env.envs import CustomActionMaskedEnvironment
import supersuit as ss
from clean_ppo import Agent  # Ensure your agent class is imported from the right file

# Define the environment setup
def create_env():
    env = CustomActionMaskedEnvironment(render_mode='human')
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    return env

# Load the trained model
env = create_env()
model_path = os.path.join(f"models/trained_model_pong_v3__clean_ppo__1__1735509872.pth")
env = ss.pettingzoo_env_to_vec_env_v1(env)

# Wrap the environment to match training setup
env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class="gymnasium")
env.single_observation_space = env.observation_space
env.single_action_space = env.action_space
env.is_vector_env = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the agent
agent = Agent(env).to(device)

# Load model weights
agent.load_state_dict(torch.load(model_path))
agent.eval()

# Run the environment loop
obs, info = env.reset(seed=1)
obs = torch.tensor(obs).to(device)

done = False
while not done:
    with torch.no_grad():
        device = next(agent.parameters()).device
        if isinstance(obs, np.ndarray):  # Check if x is a NumPy array
            obs = torch.tensor(obs)
        obs = obs.to(device)
        action, _, _, _ = agent.get_action_and_value(obs)
    obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
    done = terminated.any() or truncated.any()

    # Render the environment (if supported)
    env.render()
    print(f"Reward: {reward}")

env.close()
