import gymnasium


class ContinualWrapper(gymnasium.Wrapper):
    def __init__(self, env, change_task_every_n_episodes: int = 50):
        super().__init__(env)
        self.change_task_every_n_episodes = change_task_every_n_episodes
        self.episode_counter = 0
        self.task = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.episode_counter += 1
        if self.episode_counter % self.change_task_every_n_episodes == 0:
            self.task = (self.task + 1) % info["n_tasks"]
            self.env.reset(options={"task": self.task})
        return obs, reward, terminated, truncated, info
