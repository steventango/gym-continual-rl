from abc import ABC, abstractmethod

from gymnasium import Env


class BaseContinualEnv(Env, ABC):
    @abstractmethod
    def change_task(self, task: int) -> None:
        pass
