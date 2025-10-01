import gymnasium
from gymnasium.envs.registration import register

env_id = 'gymnasium_env/GridWorld-v0'
env_id_2 = 'gymnasium_env/RealWorld-v0'

if env_id not in gymnasium.envs.registry:
    register(
        id="gymnasium_env/GridWorld-v0",
        entry_point="gymnasium_env.envs:GridWorldEnv",
    )

if env_id_2 not in gymnasium.envs.registry:
    register(
        id="gymnasium_env/RealWorld-v0",
        entry_point="gymnasium_env.envs:RealWorldEnv",
    )
