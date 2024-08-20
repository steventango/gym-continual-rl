from pathlib import Path

import gymnasium as gym
import numpy as np
from jbw.visualizer import MapVisualizer
from PIL import Image

import gym_continual_rl  # noqa: F401
from gym_continual_rl.envs.jbw1_env import get_reward_0, get_reward_1

UP = 0
LEFT = 1
RIGHT = 2
DOWN = 3


def test_jbw_env_task_0():
    env = gym.make("gym_continual_rl/JBW-v1", task=0)
    _ = env.reset(seed=0)
    assert env.unwrapped._reward_fn == get_reward_0

    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(LEFT)
        assert reward == 0
        assert not terminated
        assert not truncated

    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(UP)
        assert reward == 0
        assert not terminated
        assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert reward == 2
    assert not terminated
    assert not truncated

    for _ in range(8):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 0
        assert not terminated
        assert not truncated

    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == -1
        assert not terminated
        assert not truncated

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 0
        assert not terminated
        assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert reward == 0.1
    assert not terminated
    assert not truncated


def test_jbw_env_task_1():
    env = gym.make("gym_continual_rl/JBW-v1", task=1)
    _ = env.reset(seed=0)
    assert env.unwrapped._reward_fn == get_reward_1

    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(LEFT)
        assert reward == 0
        assert not terminated
        assert not truncated

    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(UP)
        assert reward == 0
        assert not terminated
        assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert reward == -1
    assert not terminated
    assert not truncated

    for _ in range(8):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 0
        assert not terminated
        assert not truncated

    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 2
        assert not terminated
        assert not truncated

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 0
        assert not terminated
        assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert reward == 0.1
    assert not terminated
    assert not truncated


def test_jbw_env_render():
    env = gym.make("gym_continual_rl/JBW-v1", task=0, render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    image = Image.fromarray(frame)
    IMAGES_PATH = Path(__file__).parent.parent.parent / "images"
    expected_image = Image.open(IMAGES_PATH / "jbw.png")
    assert np.array_equal(np.array(image), np.array(expected_image))
    env.close()


def test_jbw_env_render_zoomed_out():
    env = gym.make("gym_continual_rl/JBW-v1", task=0, render_mode="rgb_array")
    env.reset(seed=0)
    env.unwrapped._painter = MapVisualizer(env.unwrapped._sim, env.unwrapped.sim_config, bottom_left=(-512, -512), top_right=(512, 512))

    for i in range(1, 8):
        side_length = i * 128
        for _ in range(side_length // 2):
            obs, reward, terminated, truncated, info = env.step(UP)

        for _ in range(side_length // 2):
            obs, reward, terminated, truncated, info = env.step(RIGHT)

        for _ in range(side_length):
            obs, reward, terminated, truncated, info = env.step(DOWN)

        for _ in range(side_length):
            obs, reward, terminated, truncated, info = env.step(LEFT)

        for _ in range(side_length):
            obs, reward, terminated, truncated, info = env.step(UP)

        for _ in range(side_length // 2):
            obs, reward, terminated, truncated, info = env.step(RIGHT)

        for _ in range(side_length // 2):
            obs, reward, terminated, truncated, info = env.step(DOWN)

    frame = env.render()
    image = Image.fromarray(frame)
    IMAGES_PATH = Path(__file__).parent.parent.parent / "images"
    expected_image = Image.open(IMAGES_PATH / "jbw_zoomed_out.png")
    assert np.array_equal(np.array(image), np.array(expected_image))
    env.close()
