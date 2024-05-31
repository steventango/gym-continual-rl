from gymnasium.envs.registration import register

register(
    id="gym_continual_rl/GridWorld-v0",
    entry_point="gym_continual_rl.envs:GridWorldEnv",
)
