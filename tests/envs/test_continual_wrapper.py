import gymnasium as gym

from gym_continual_rl.wrappers import EpisodicContinualWrapper

RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3


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
        reward = env.step(UP)[1]
    for _ in range(5):
        reward = env.step(RIGHT)[1]
    return reward


def run_g2_trajectory(env: gym.Env):
    for _ in range(5):
        reward = env.step(RIGHT)[1]
    for _ in range(4):
        reward = env.step(UP)[1]
    return reward


def test_continual_wrapper():
    raise NotImplementedError
