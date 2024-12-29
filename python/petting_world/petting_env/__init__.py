from gymnasium.envs.registration import register

register(
    id="petting_env/GridWorld-v0",
    entry_point="petting_env.envs:GridWorldEnv",
)
