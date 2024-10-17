import gymnasium
from gymnasium.envs.registration import register

env_id = 'gymnasium_env/GridWorld-v0'

if env_id not in gymnasium.envs.registry:
    register(
        id="gymnasium_env/GridWorld-v0",
        entry_point="gymnasium_env.envs:GridWorldEnv",
    )
