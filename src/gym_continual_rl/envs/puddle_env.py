import json

import numpy as np
from gym_puddle.env.puddle_env import PuddleEnv as BasePuddleEnv

from gym_continual_rl.envs.base import BaseContinualEnv


class PuddleEnv(BaseContinualEnv, BasePuddleEnv):
    def __init__(
        self,
        *args,
        env_setups: list[dict] = [],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not env_setups:
            self.env_setups = []
            for i in range(1, 6):
                with open(f"gym-puddle/gym_puddle/env_configs/pw{i}.json") as f:
                    env_setup = json.load(f)
                self.env_setups.append(env_setup)

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

    def change_task(self, task: int) -> None:
        env_setup = self.env_setups[task]
        self.change(
            start=env_setup["start"],
            goal=env_setup["goal"],
            goal_threshold=env_setup["goal_threshold"],
            noise=env_setup["noise"],
            thrust=env_setup["thrust"],
            puddle_top_left=env_setup["puddle_top_left"],
            puddle_width=env_setup["puddle_width"],
        )
