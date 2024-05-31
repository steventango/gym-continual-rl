import gymnasium as gym
import numpy as np

import gym_continual_rl

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

    # RIGHT
    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([2, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # UP
    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # LEFT
    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # UP
    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # RIGHT
    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([3, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # UP
    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([3, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # LEFT
    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([3, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # RIGHT, RIGHT, RIGHT
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

    # RIGHT
    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([2, 1]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # UP
    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # LEFT
    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # UP
    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # RIGHT
    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([3, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # UP
    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([4, 2]))  # Slips RIGHT
    assert reward == 0
    assert not terminated
    assert not truncated

    # LEFT
    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([3, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # UP
    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([2, 2]))  # Slips LEFT
    assert reward == 0
    assert not terminated
    assert not truncated

    # RIGHT
    obs, reward, terminated, truncated, info = env.step(RIGHT)
    assert np.array_equal(obs, np.array([3, 2]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # UP
    obs, reward, terminated, truncated, info = env.step(UP)
    assert np.array_equal(obs, np.array([3, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # LEFT
    obs, reward, terminated, truncated, info = env.step(LEFT)
    assert np.array_equal(obs, np.array([3, 3]))
    assert reward == 0
    assert not terminated
    assert not truncated

    # RIGHT, RIGHT, RIGHT
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
    expected_frame = np.load("tests/envs/test_grid_world_render.npy")
    assert np.array_equal(frame, expected_frame)
    env.close()
