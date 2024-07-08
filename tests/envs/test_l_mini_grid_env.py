from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import gym_continual_rl  # noqa: F401


def test_l_mini_grid_env_task0_goal0():
    env = gym.make("gym_continual_rl/LMiniGrid-v0", task=0)
    obs, info = env.reset(seed=0)
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.right)
    assert obs is not None
    assert reward == 0
    assert not terminated
    assert not truncated
    for _ in range(8):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.right)
    assert obs is not None
    assert reward == 0
    assert not terminated
    assert not truncated
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.forward)
    assert obs is not None
    assert reward == 5
    assert terminated
    assert not truncated

def test_l_mini_grid_env_task0_goal1():
    env = gym.make("gym_continual_rl/LMiniGrid-v0", task=0)
    obs, info = env.reset(seed=0)
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.right)
    assert obs is not None
    assert reward == 0
    assert not terminated
    assert not truncated
    for _ in range(6):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.right)
    assert obs is not None
    assert reward == 0
    assert not terminated
    assert not truncated
    for _ in range(6):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.forward)
    assert obs is not None
    assert reward == 0
    assert terminated
    assert not truncated

def test_l_mini_grid_env_task1_goal0():
    env = gym.make("gym_continual_rl/LMiniGrid-v0", task=1)
    obs, info = env.reset(seed=0)
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.right)
    assert obs is not None
    assert reward == 0
    assert not terminated
    assert not truncated
    for _ in range(8):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.right)
    assert obs is not None
    assert reward == 0
    assert not terminated
    assert not truncated
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.forward)
    assert obs is not None
    assert reward == 0
    assert terminated
    assert not truncated


def test_l_mini_grid_env_task1_goal1():
    env = gym.make("gym_continual_rl/LMiniGrid-v0", task=1)
    obs, info = env.reset(seed=0)
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.right)
    assert obs is not None
    assert reward == 0
    assert not terminated
    assert not truncated
    for _ in range(6):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.right)
    assert obs is not None
    assert reward == 0
    assert not terminated
    assert not truncated
    for _ in range(6):
        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
        assert obs is not None
        assert reward == 0
        assert not terminated
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(env.actions.forward)
    assert obs is not None
    assert reward == 5
    assert terminated
    assert not truncated


def test_l_mini_grid_render():
    env = gym.make("gym_continual_rl/LMiniGrid-v0", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    image = Image.fromarray(frame)
    IMAGES_PATH = Path(__file__).parent.parent.parent / "images"
    expected_image = Image.open(IMAGES_PATH / "3.c.png")
    assert np.array_equal(np.array(image), np.array(expected_image))
    env.close()
