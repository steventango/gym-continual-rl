from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

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


def test_jbw_env_render():
    env = gym.make("gym_continual_rl/JBW-v0", task=0, render=True)
    env.reset(seed=0)
    frame = env.render()
    image = Image.fromarray(frame)
    IMAGES_PATH = Path(__file__).parent.parent.parent / "images"
    expected_image = Image.open(IMAGES_PATH / "jbw.png")
    assert np.array_equal(np.array(image), np.array(expected_image))
    env.close()
