import gym_continual_rl
import gymnasium as gym
import numpy as np


def test_grid_world():
    env = gym.make("gym_continual_rl/GridWorld-v0")
    obs, info = env.reset(seed=0)
    assert obs == [0, 0]
