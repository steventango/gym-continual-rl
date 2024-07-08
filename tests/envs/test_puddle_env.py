import gymnasium as gym
import numpy as np

import gym_continual_rl  # noqa: F401


def test_puddle_env_task_0():
    env = gym.make("gym_continual_rl/PuddleWorld-v0", task=0)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[0]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[0]["puddle_width"])


def test_puddle_env_task_1():
    env = gym.make("gym_continual_rl/PuddleWorld-v0", task=1)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[1]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[1]["puddle_width"])


def test_puddle_env_task_2():
    env = gym.make("gym_continual_rl/PuddleWorld-v0", task=2)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[2]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[2]["puddle_width"])


def test_puddle_env_task_3():
    env = gym.make("gym_continual_rl/PuddleWorld-v0", task=3)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[3]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[3]["puddle_width"])


def test_puddle_env_task_4():
    env = gym.make("gym_continual_rl/PuddleWorld-v0", task=4)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[4]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[4]["puddle_width"])


CUSTOM_ENV_SETUPS = [
    {
        "start": [0.1, 0.1],
        "goal": [0.9, 0.9],
        "goal_threshold": 0.1,
        "noise": 0.01,
        "thrust": 0.05,
        "puddle_top_left": [[0.2, 0.8]],
        "puddle_width": [[0.6, 0.6]],
    },
    {
        "start": [0.9, 0.9],
        "goal": [0.1, 0.1],
        "goal_threshold": 0.1,
        "noise": 0.01,
        "thrust": 0.05,
        "puddle_top_left": [[0.2, 0.8]],
        "puddle_width": [[0.6, 0.6]],
    },
]


def test_puddle_env_custom_task_0():
    env = gym.make("gym_continual_rl/PuddleWorld-v0", render_mode="rgb_array", env_setups=CUSTOM_ENV_SETUPS, task=0)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.start, env.unwrapped.env_setups[0]["start"])
    assert np.array_equal(env.unwrapped.goal, env.unwrapped.env_setups[0]["goal"])

def test_puddle_env_custom_task_1():
    env = gym.make("gym_continual_rl/PuddleWorld-v0", render_mode="rgb_array", env_setups=CUSTOM_ENV_SETUPS, task=1)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.start, env.unwrapped.env_setups[1]["start"])
    assert np.array_equal(env.unwrapped.goal, env.unwrapped.env_setups[1]["goal"])
