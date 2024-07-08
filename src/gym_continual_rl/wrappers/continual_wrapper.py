from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.utils import seeding

from gym_continual_rl.envs.base import BaseContinualEnv


class ContinualWrapper(gym.Wrapper):
    def __init__(
        self,
        env: BaseContinualEnv,
        task_duration: int = 50,
        n_tasks: int = 2,
        sample: bool = False,
        seed: int | None = None,
    ):
        super().__init__(env)
        self.task_duration = task_duration
        self.counter = 0
        self.task = 0
        self.n_tasks = n_tasks
        self.sample = sample
        self.np_random, seed = seeding.np_random(seed)
        if self.sample:
            self.change_task()

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.counter += 1
        if self.counter % self.task_duration == 0:
            self.change_task()
        return obs, reward, terminated, truncated, info

    def change_task(self):
        if self.sample:
            self.task = self.np_random.integers(self.n_tasks)
        else:
            self.task = (self.task + 1) % self.n_tasks
        self.env.change_task(self.task)


class EpisodicContinualWrapper(ContinualWrapper):
    def __init__(
        self,
        env: BaseContinualEnv,
        task_duration: int = 50,
        n_tasks: int = 2,
        sample: bool = False,
        seed: int | None = None,
    ):
        super().__init__(env, task_duration, n_tasks, sample, seed)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.counter += 1
            if self.counter % self.task_duration == 0:
                self.change_task()
        return obs, reward, terminated, truncated, info
