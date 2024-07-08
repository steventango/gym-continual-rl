import gymnasium as gym
import numpy as np

import gym_continual_rl  # noqa: F401
from gym_continual_rl.envs.grid_world import Action
from gym_continual_rl.wrappers import ContinualWrapper, EpisodicContinualWrapper


def test_episodic_continual_wrapper_g1_trajectory():
    env = gym.make("gym_continual_rl/GridWorld-v0", slippery=0.0)
    env = EpisodicContinualWrapper(env, task_duration=50)
    env.reset()

    for _ in range(50):
        assert env.task == 0
        reward = run_g1_trajectory(env)
        assert reward == 1
        env.reset()

    for _ in range(50):
        assert env.task == 1
        reward = run_g1_trajectory(env)
        assert reward == -1
        env.reset()

    assert env.task == 0
    reward = run_g1_trajectory(env)
    assert reward == 1
    assert env.task == 0

    env.close()


def test_episodic_continual_wrapper_g2_trajectory():
    env = gym.make("gym_continual_rl/GridWorld-v0", slippery=0.0)
    env = EpisodicContinualWrapper(env, task_duration=50)
    env.reset()

    for _ in range(50):
        assert env.unwrapped.task == 0
        reward = run_g2_trajectory(env)
        assert reward == -1
        env.reset()

    for _ in range(50):
        assert env.unwrapped.task == 1
        reward = run_g2_trajectory(env)
        assert reward == 1
        env.reset()

    assert env.unwrapped.task == 0
    reward = run_g2_trajectory(env)
    assert reward == -1
    assert env.unwrapped.task == 0

    env.close()


def run_g1_trajectory(env: gym.Env):
    for _ in range(5):
        reward = env.step(Action.UP)[1]
    for _ in range(5):
        reward = env.step(Action.RIGHT)[1]
    return reward


def run_g2_trajectory(env: gym.Env):
    for _ in range(5):
        reward = env.step(Action.RIGHT)[1]
    for _ in range(4):
        reward = env.step(Action.UP)[1]
    return reward


def test_continual_wrapper():
    env = gym.make("gym_continual_rl/Puddle-v0")
    env = ContinualWrapper(env, task_duration=1, n_tasks=5)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[0]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[0]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[1]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[1]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[2]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[2]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[3]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[3]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[4]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[4]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[0]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[0]["puddle_width"])

    env.close()


def test_continual_wrapper_sample():
    env = gym.make("gym_continual_rl/Puddle-v0")
    env = ContinualWrapper(env, task_duration=1, n_tasks=5, sample=True, seed=0)
    _ = env.reset(seed=0)
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[4]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[4]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[3]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[3]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[2]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[2]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[1]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[1]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[1]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[1]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[0]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[0]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[0]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[0]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[0]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[0]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[0]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[0]["puddle_width"])

    env.step(env.action_space.sample())
    assert np.array_equal(env.unwrapped.puddle_top_left, env.unwrapped.env_setups[4]["puddle_top_left"])
    assert np.array_equal(env.unwrapped.puddle_width, env.unwrapped.env_setups[4]["puddle_width"])

    env.close()
