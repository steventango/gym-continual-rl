import gymnasium as gym
import numpy as np

import gym_continual_rl  # noqa: F401
from gym_continual_rl.envs.jbw_env import get_reward_0, get_reward_1


def test_jbw_env_task_0():
    env = gym.make("gym_continual_rl/JBW-v0", task=0)
    _ = env.reset(seed=0)
    assert env.unwrapped._reward_fn == get_reward_0


def test_jbw_env_task_1():
    env = gym.make("gym_continual_rl/JBW-v0", task=1)
    _ = env.reset(seed=0)
    assert env.unwrapped._reward_fn == get_reward_1

