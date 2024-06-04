from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import gym_continual_rl  # noqa: F401

RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3


def test_grid_world_no_slippery_g1():
    env = gym.make("gym_continual_rl/GridWorld-v0", slippery=0.0)
    obs, info = env.reset(seed=0)
    assert np.array_equal(obs, np.array([0, 0]))

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([0, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([0, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([0, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([0, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([0, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([0, 4]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([1, 4]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert np.array_equal(obs, np.array([1, 4]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([2, 4]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert np.array_equal(obs, np.array([2, 4]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 5]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([3, 5]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([3, 5]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([4, 5]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([5, 5]))
    assert reward == 1
    assert terminated
    assert not truncated

    env.close()


def test_grid_world_no_slippery_g2():
    env = gym.make("gym_continual_rl/GridWorld-v0", slippery=0.0)
    obs, info = env.reset(seed=0)
    assert np.array_equal(obs, np.array([0, 0]))

    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([0, 0]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert np.array_equal(obs, np.array([0, 0]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([1, 0]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([1, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([1, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([2, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([3, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([3, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([3, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([4, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([5, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([5, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([5, 4]))
    assert reward == -1
    assert terminated
    assert not truncated

    env.close()


def test_grid_world_no_slippery_g1_task2():
    env = gym.make("gym_continual_rl/GridWorld-v0", slippery=0.0)
    obs, info = env.reset(seed=0, options={"task": 1})
    assert np.array_equal(obs, np.array([0, 0]))

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([0, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([0, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([0, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([0, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([0, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([0, 4]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([1, 4]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert np.array_equal(obs, np.array([1, 4]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([2, 4]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert np.array_equal(obs, np.array([2, 4]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 5]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([3, 5]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([3, 5]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([4, 5]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([5, 5]))
    assert reward == -1
    assert terminated
    assert not truncated

    env.close()


def test_grid_world_no_slippery_g2_tasks2():
    env = gym.make("gym_continual_rl/GridWorld-v0", slippery=0.0)
    obs, info = env.reset(seed=0, options={"task": 1})
    assert np.array_equal(obs, np.array([0, 0]))

    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([0, 0]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert np.array_equal(obs, np.array([0, 0]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([1, 0]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([1, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([1, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([2, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([3, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([3, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([3, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([4, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([5, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([5, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([5, 4]))
    assert reward == 1
    assert terminated
    assert not truncated

    env.close()


def test_grid_world_slippery():
    env = gym.make("gym_continual_rl/GridWorld-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    assert np.array_equal(obs, np.array([0, 0]))

    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([0, 0]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert np.array_equal(obs, np.array([0, 0]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([0, 1]))  # Slips UP
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([1, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([1, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([2, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([3, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([4, 2]))  # Slips RIGHT
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([3, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))  # Slips LEFT
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([3, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([3, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([3, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([4, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([5, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([5, 2]))  # Slips DOWN
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([5, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([5, 4]))
    assert reward == -1
    assert terminated
    assert not truncated

    env.close()


def test_grid_world_render():
    env = gym.make("gym_continual_rl/GridWorld-v0", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    image = Image.fromarray(frame)
    IMAGES_PATH = Path(__file__).parent.parent.parent / "images"
    expected_image = Image.open(IMAGES_PATH / "3.a.png")
    assert np.array_equal(np.array(image), np.array(expected_image))
    env.close()
