from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import gym_continual_rl  # noqa: F401
from gym_continual_rl.envs.puddle_env import Action


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

PUDDLE_WIDTH = 0.4
CUSTOM_ENV_SETUPS = [
    {
        "start": [0.1, 0.1],
        "goal": [0.9, 0.9],
        "goal_threshold": 0.1,
        "noise": 0.02,
        "thrust": 0.1,
        "puddle_top_left": [[(1 - PUDDLE_WIDTH) / 2, 1 - (1 - PUDDLE_WIDTH) / 2]],
        "puddle_width": [[PUDDLE_WIDTH, PUDDLE_WIDTH]],
    },
    {
        "start": [0.9, 0.9],
        "goal": [0.1, 0.1],
        "goal_threshold": 0.1,
        "noise": 0.02,
        "thrust": 0.1,
        "puddle_top_left": [[(1 - PUDDLE_WIDTH) / 2, 1 - (1 - PUDDLE_WIDTH) / 2]],
        "puddle_width": [[PUDDLE_WIDTH, PUDDLE_WIDTH]],
    },
]


def test_puddle_env_custom_task_0():
    env = gym.make("gym_continual_rl/PuddleWorld-v0", env_setups=CUSTOM_ENV_SETUPS, task=0)
    env.reset(seed=0)
    assert np.array_equal(env.unwrapped.start, env.unwrapped.env_setups[0]["start"])
    assert np.array_equal(env.unwrapped.goal, env.unwrapped.env_setups[0]["goal"])


def test_puddle_env_custom_task_1():
    env = gym.make("gym_continual_rl/PuddleWorld-v0", env_setups=CUSTOM_ENV_SETUPS, task=1)
    env.reset(seed=0)
    assert np.array_equal(env.unwrapped.start, env.unwrapped.env_setups[1]["start"])
    assert np.array_equal(env.unwrapped.goal, env.unwrapped.env_setups[1]["goal"])


def test_puddle_env_custom_task_0_direct_trajectory():
    G = 0
    env = gym.make("gym_continual_rl/PuddleWorld-v0", env_setups=CUSTOM_ENV_SETUPS, task=0)
    env.reset(seed=0)
    total_steps = 0
    for i in range(100):
        action = Action.RIGHT if i % 2 == 0 else Action.UP
        obs, reward, done, trunc, info = env.step(action)
        G += reward
        total_steps += 1
        if done or trunc:
            break
    assert G == -1004.7029543263184
    assert total_steps == 17


def test_puddle_env_custom_task_0_indirect_trajectory():
    G = 0
    env = gym.make("gym_continual_rl/PuddleWorld-v0", env_setups=CUSTOM_ENV_SETUPS, task=0)
    env.reset(seed=0)
    for _ in range(8):
        action = Action.UP
        obs, reward, done, trunc, info = env.step(action)
        G += reward
        assert not done
        assert not trunc
    for _ in range(8):
        action = Action.RIGHT
        obs, reward, done, trunc, info = env.step(action)
        G += reward
        assert not done
        assert not trunc
    # extra step to adjust for the noise
    obs, reward, done, trunc, info = env.step(action)
    G += reward
    assert done
    assert G == -16
    # without noise, the optimal trajectory would have G = -15


def test_puddle_env_custom_task_1_direct_trajectory():
    G = 0
    env = gym.make("gym_continual_rl/PuddleWorld-v0", env_setups=CUSTOM_ENV_SETUPS, task=1)
    env.reset(seed=0)
    total_steps = 0
    for i in range(16):
        action = Action.LEFT if i % 2 == 0 else Action.DOWN
        obs, reward, done, trunc, info = env.step(action)
        G += reward
        total_steps += 1
    # note it is possible to go past the goal
    # extra step to adjust for the noise
    obs, reward, done, trunc, info = env.step(action.RIGHT)
    G += reward
    total_steps += 1
    assert done
    assert G == -657.5067921677007
    assert total_steps == 17


def test_puddle_env_custom_task_1_indirect_trajectory():
    G = 0
    env = gym.make("gym_continual_rl/PuddleWorld-v0", env_setups=CUSTOM_ENV_SETUPS, task=1)
    env.reset(seed=0)
    for _ in range(8):
        action = Action.DOWN
        obs, reward, done, trunc, info = env.step(action)
        G += reward
        assert not done
        assert not trunc
    for _ in range(6):
        action = Action.LEFT
        obs, reward, done, trunc, info = env.step(action)
        G += reward
        assert not done
        assert not trunc
    obs, reward, done, trunc, info = env.step(action)
    G += reward
    assert done
    assert G == -14
    # without noise, the optimal trajectory would have G = -15


def test_puddle_env_render():
    env = gym.make("gym_continual_rl/PuddleWorld-v0", render_mode="rgb_array", env_setups=CUSTOM_ENV_SETUPS, task=0)
    env.reset(seed=0)
    frame = env.render()
    image = Image.fromarray(frame)
    IMAGES_PATH = Path(__file__).parent.parent.parent / "images"
    expected_image = Image.open(IMAGES_PATH / "puddle_world.png")
    assert np.array_equal(np.array(image), np.array(expected_image))
    env.close()
