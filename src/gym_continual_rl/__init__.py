from gymnasium.envs.registration import register

register(
    id="gym_continual_rl/GridWorld-v0",
    entry_point="gym_continual_rl.envs:GridWorldEnv",
)

register(
    id="gym_continual_rl/LMiniGrid-v0",
    entry_point="gym_continual_rl.envs:LMiniGridEnv",
)

register(
    id="gym_continual_rl/PuddleWorld-v0",
    entry_point="gym_continual_rl.envs:PuddleEnv",
)

register(
    id="gym_continual_rl/JBW-v0",
    entry_point="gym_continual_rl.envs:JBWEnv",
    kwargs={
        "f_type": "obj",
    },
)
