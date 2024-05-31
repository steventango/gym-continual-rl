import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str = None,
        size: int = 6,
        start_location: np.ndarray = None,
        goal_locations: list[np.ndarray] = None,
        goal_rewards: list[np.ndarray] = None,
        obstacles: list[np.ndarray] = None,
        slippery: float = 0.1,
    ):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations is the agent's location.
        # Encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]), # right
            1: np.array([0, 1]), # up
            2: np.array([-1, 0]), # left
            3: np.array([0, -1]), # down
        }
        self.start_location = start_location if start_location is not None else np.array([0, 0])
        self.goal_locations = (
            goal_locations
            if goal_locations is not None
            else [
                np.array([size - 1, size - 1]),
                np.array([size - 1, size - 2]),
            ]
        )
        self.goal_rewards = goal_rewards if goal_rewards is not None else [1, -1]
        self.task = 0
        self.obstacles = obstacles if obstacles is not None else [
            np.array([1, 2]),
            np.array([1, 3]),
            np.array([2, 3]),
        ]
        self.slippery = slippery

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {
            "n_tasks": len(self.goal_locations),
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if options is not None:
            self.task = options["task"]

        self._agent_location = self.start_location

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # The action has the intended effect (1-self.slippery)% of the time;
        # otherwise, it is swapped with one of the perpendicular actions
        if self.np_random.uniform() < self.slippery:
            action = self.np_random.choice([0, 2] if action % 2 else [1, 3])
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        if all(not np.array_equal(new_location, obstacle) for obstacle in self.obstacles):
            self._agent_location = new_location

        # An episode is done iff the agent has reached any of the goal locations
        reward = 0
        for goal_location, goal_reward in zip(self.goal_locations, self.goal_rewards):
            if np.array_equal(self._agent_location, goal_location):
                reward = goal_reward
                break
        done = reward != 0
        if self.task == 1:
            reward = -reward
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size  # The size of a single grid square in pixels

        # First we draw the target
        for i, goal_location in enumerate(self.goal_locations):
            goal_location = goal_location.copy()
            goal_location[1] = self.size - 1 - goal_location[1]
            pygame.draw.rect(
                canvas,
                (131, 193, 103),
                pygame.Rect(
                    pix_square_size * goal_location,
                    (pix_square_size, pix_square_size),
                ),
            )
            # add labels to goals
            font = pygame.font.SysFont("dejavuserif", 64)
            text = font.render(f"G{i+1}", True, (0, 0, 0))
            text_rect = text.get_rect(center=(pix_square_size * goal_location + pix_square_size / 2))
            canvas.blit(text, text_rect)

        # Next we draw the obstacles
        for obstacle in self.obstacles:
            obstacle = obstacle.copy()
            obstacle[1] = self.size - 1 - obstacle[1]
            pygame.draw.rect(
                canvas,
                (136, 136, 136),
                pygame.Rect(
                    pix_square_size * obstacle,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent
        agent_location = self._agent_location.copy()
        agent_location[1] = self.size - 1 - agent_location[1]
        agent_center = (agent_location + 0.5) * pix_square_size
        agent_size = pix_square_size / 3
        agent_points = [
            (agent_center[0], agent_center[1] - agent_size),
            (agent_center[0] - agent_size, agent_center[1] + agent_size),
            (agent_center[0] + agent_size, agent_center[1] + agent_size),
        ]
        pygame.draw.polygon(canvas, (255, 0, 0), agent_points)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
