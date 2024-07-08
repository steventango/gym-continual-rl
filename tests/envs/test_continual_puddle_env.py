import numpy as np

from src.continual_puddle_env import ContinualPuddleEnv

PUDDLE_TOP_LEFT_PW1 = np.array([[0, 0.85], [0.35, 0.9]])
PUDDLE_WIDTH_PW1 = np.array([[0.55, 0.2], [0.2, 0.6]])
PUDDLE_TOP_LEFT_PW2 = np.array([[0, 0.85], [0.6, 0.9]])
PUDDLE_WIDTH_PW2 = np.array([[0.55, 0.2], [0.2, 0.6]])
PUDDLE_TOP_LEFT_PW3 = np.array([[0.8, 0.85], [0.35, 0.9]])
PUDDLE_WIDTH_PW3 = np.array([[0.2, 0.2], [0.2, 0.6]])
PUDDLE_TOP_LEFT_PW4 = np.array([[0.2, 0.85], [0.2, 0.35], [0.6, 0.75]])
PUDDLE_WIDTH_PW4 = np.array([[0.5, 0.2], [0.5, 0.2], [0.2, 0.5]])
PUDDLE_TOP_LEFT_PW5 = np.array(
    [[0.1, 0.85], [0.1, 0.25], [0.5, 0.85], [0.5, 0.35], [0.3, 0.6]]
)
PUDDLE_WIDTH_PW5 = np.array(
    [[0.3, 0.2], [0.3, 0.2], [0.2, 0.3], [0.2, 0.3], [0.2, 0.3]]
)


def test_continual_puddle_env():
    env = ContinualPuddleEnv(change_every_timesteps=1)
    _ = env.reset(seed=0)
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW1)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW1)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW2)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW2)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW3)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW3)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW4)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW4)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW5)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW5)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW1)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW1)

    env.close()


def test_continual_puddle_env_sample():
    env = ContinualPuddleEnv(change_every_timesteps=1, sample=True, seed=0)
    _ = env.reset(seed=0)
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW5)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW5)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW4)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW4)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW3)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW3)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW2)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW2)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW2)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW2)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW1)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW1)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW1)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW1)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW1)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW1)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW1)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW1)

    env.step(env.action_space.sample())
    assert np.all(env.puddle_top_left == PUDDLE_TOP_LEFT_PW5)
    assert np.all(env.puddle_width == PUDDLE_WIDTH_PW5)

    env.close()
