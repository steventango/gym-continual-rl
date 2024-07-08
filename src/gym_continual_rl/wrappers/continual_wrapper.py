from typing import Any

import gymnasium as gym
from gymnasium.core import WrapperObsType


class EpisodicContinualWrapper(gym.Wrapper):
    def __init__(self, env, change_task_every_n_episodes: int = 50, n_tasks: int = 2):
        super().__init__(env)
        self.change_task_every_n_episodes = change_task_every_n_episodes
        self.episode_counter = 0
        self.task = 0
        self.n_tasks = n_tasks

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.episode_counter += 1
            if self.episode_counter % self.change_task_every_n_episodes == 0:
                self.task = (self.task + 1) % self.n_tasks
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        if options is None:
            options = {}
        options["task"] = self.task
        return self.env.reset(seed=seed, options=options)
