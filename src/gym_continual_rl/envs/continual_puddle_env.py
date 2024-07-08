import json

import numpy as np
from gym_puddle.env.puddle_env import PuddleEnv
from gymnasium.utils import seeding


class ContinualPuddleEnv(PuddleEnv):
    def __init__(
        self,
        *args,
        change_every_timesteps: int = 150_000,
        sample: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_setups = []
        for i in range(1, 6):
            with open(f"gym-puddle/gym_puddle/env_configs/pw{i}.json") as f:
                env_setup = json.load(f)
            self.env_setups.append(env_setup)
        self.change_every_timesteps = change_every_timesteps
        self.global_num_steps = 0
        self.global_np_random, seed = seeding.np_random(seed)
        self.sample = sample
        self.env_setup_index = 0
        if sample:
            self.env_setup_index = self.sample_env_setup_index()
            self.change_by_index(self.env_setup_index)

    def change(
        self,
        start: list[float] = [0.2, 0.4],
        goal: list[float] = [1.0, 1.0],
        goal_threshold: float = 0.1,
        noise: float = 0.01,
        thrust: float = 0.05,
        puddle_top_left: list[list[float]] = [[0, 0.85], [0.35, 0.9]],
        puddle_width: list[list[float]] = [[0.55, 0.2], [0.2, 0.6]],
    ) -> None:
        self.start = np.array(start)
        self.goal = np.array(goal)

        self.goal_threshold = goal_threshold

        self.noise = noise
        self.thrust = thrust

        self.puddle_top_left = [np.array(top_left) for top_left in puddle_top_left]
        self.puddle_width = [np.array(width) for width in puddle_width]

        self.actions = [np.zeros(2) for i in range(4)]

        for i in range(4):
            self.actions[i][i // 2] = thrust * (i % 2 * 2 - 1)

    def change_by_index(self, index: int) -> None:
        env_setup = self.env_setups[index]
        self.change(
            start=env_setup["start"],
            goal=env_setup["goal"],
            goal_threshold=env_setup["goal_threshold"],
            noise=env_setup["noise"],
            thrust=env_setup["thrust"],
            puddle_top_left=env_setup["puddle_top_left"],
            puddle_width=env_setup["puddle_width"],
        )

    def sample_env_setup_index(self) -> int:
        return self.global_np_random.integers(len(self.env_setups))

    def step(self, action):
        self.global_num_steps += 1
        if self.global_num_steps % self.change_every_timesteps == 0:
            if self.sample:
                self.env_setup_index = self.sample_env_setup_index()
            else:
                self.env_setup_index = (
                    self.global_num_steps // self.change_every_timesteps
                ) % len(self.env_setups)
            self.change_by_index(self.env_setup_index)
        return super().step(action)
