import gymnasium as gym

from gym_continual_rl.wrappers import ContinualWrapper

RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3


def test_continual_wrapper_g1_trajectory():
    env = gym.make("gym_continual_rl/GridWorld-v0", slippery=0.0)
    env = ContinualWrapper(env, change_task_every_n_episodes=50)
    env.reset()

    for _ in range(50):
        reward = run_g1_trajectory(env)
        assert reward == 1
        assert env.unwrapped.task == 0
        env.reset()

    for _ in range(50):
        reward = run_g1_trajectory(env)
        assert reward == -1
        assert env.unwrapped.task == 1
        env.reset()

    reward = run_g1_trajectory(env)
    assert reward == 1
    assert env.unwrapped.task == 0

    env.close()


def test_continual_wrapper_g2_trajectory():
    env = gym.make("gym_continual_rl/GridWorld-v0", slippery=0.0)
    env = ContinualWrapper(env, change_task_every_n_episodes=50)
    env.reset()

    for _ in range(50):
        reward = run_g2_trajectory(env)
        assert reward == -1
        assert env.unwrapped.task == 0
        env.reset()

    for _ in range(50):
        reward = run_g2_trajectory(env)
        assert reward == 1
        assert env.unwrapped.task == 1
        env.reset()

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
