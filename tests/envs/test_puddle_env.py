import gymnasium as gym
import numpy as np

import gym_continual_rl  # noqa: F401


def test_puddle_env_task_0():
    env = gym.make("gym_continual_rl/Puddle-v0", task=0)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[0]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[0]["puddle_width"])


def test_puddle_env_task_1():
    env = gym.make("gym_continual_rl/Puddle-v0", task=1)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[1]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[1]["puddle_width"])

def test_puddle_env_task_2():
    env = gym.make("gym_continual_rl/Puddle-v0", task=2)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[2]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[2]["puddle_width"])

def test_puddle_env_task_3():
    env = gym.make("gym_continual_rl/Puddle-v0", task=3)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[3]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[3]["puddle_width"])

def test_puddle_env_task_4():
    env = gym.make("gym_continual_rl/Puddle-v0", task=4)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[4]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[4]["puddle_width"])
