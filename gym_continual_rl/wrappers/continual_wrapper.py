from typing import Any

import gymnasium as gym
from gymnasium.core import WrapperObsType


class ContinualWrapper(gym.Wrapper):
    def __init__(self, env, change_task_every_n_episodes: int = 50):
        super().__init__(env)
        self.change_task_every_n_episodes = change_task_every_n_episodes
        self.episode_counter = 0
        self.task = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            self.episode_counter += 1
            if self.episode_counter % self.change_task_every_n_episodes == 0:
                self.task = (self.task + 1) % info["n_tasks"]
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        if options is None:
            options = {}
        options["task"] = self.task
        return self.env.reset(seed=seed, options=options)
