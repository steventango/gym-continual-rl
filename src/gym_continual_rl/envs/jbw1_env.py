import gc

import gymnasium as gym
import jbw
import numpy as np
from gymnasium import logger, spaces
from jbw.agent import Agent
from jbw.direction import RelativeDirection
from jbw.item import *
from jbw.simulator import *
from jbw.visualizer import MapVisualizer, pi

from gym_continual_rl.envs.base import BaseContinualEnv

AGENT_VIEW = 5


def make_config():
    # specify the item types
    items = []
    items.append(
        Item(
            "Onion",
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0, 0, 0],
            [0, 0, 0],
            False,
            0.0,
            intensity_fn=IntensityFunction.CONSTANT,
            intensity_fn_args=[-3.5],
            interaction_fns=[
                [InteractionFunction.PIECEWISE_BOX, 3, 10, 1, -2],
                [InteractionFunction.ZERO],
                [InteractionFunction.PIECEWISE_BOX, 25, 50, -50, -10],
            ],
        )
    )
    items.append(
        Item(
            "Banana",
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0, 0, 0],
            [0, 0, 0],
            False,
            0.0,
            intensity_fn=IntensityFunction.CONSTANT,
            intensity_fn_args=[-6.0],
            interaction_fns=[[InteractionFunction.ZERO], [InteractionFunction.ZERO], [InteractionFunction.ZERO]],
        )
    )
    items.append(
        Item(
            "JellyBean",
            [0, 0, 1.0],
            [0, 0, 1.0],
            [0, 0, 0],
            [0, 0, 0],
            False,
            0.0,
            intensity_fn=IntensityFunction.CONSTANT,
            intensity_fn_args=[-3.5],
            interaction_fns=[
                [InteractionFunction.PIECEWISE_BOX, 25, 50, -50, -10],
                [InteractionFunction.ZERO],
                [InteractionFunction.PIECEWISE_BOX, 3, 10, 1, -2],
            ],
        )
    )

    # construct the simulator configuration
    return SimulatorConfig(
        max_steps_per_movement=1,
        vision_range=AGENT_VIEW,
        allowed_movement_directions=[
            ActionPolicy.ALLOWED,
            ActionPolicy.ALLOWED,
            ActionPolicy.ALLOWED,
            ActionPolicy.ALLOWED,
        ],
        allowed_turn_directions=[
            ActionPolicy.DISALLOWED,
            ActionPolicy.DISALLOWED,
            ActionPolicy.DISALLOWED,
            ActionPolicy.DISALLOWED,
        ],
        no_op_allowed=False,
        patch_size=32,
        mcmc_num_iter=4000,
        items=items,
        agent_color=[0.0, 0.0, 0.0],
        agent_field_of_view=2 * pi,
        collision_policy=MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
        decay_param=0.0,
        diffusion_param=0.14,
        deleted_item_lifetime=500,
    )


class JBW1Env(gym.Env):
    metadata = {"render_modes": ["matplotlib", "rgb_array"], "render_fps": 4}
    """
    JBW environment for OpenAI gym.
    The action space consists of four actions:
    - `0`: Move forward.
    - `1`: Move left.
    - `2`: Move right.
    - `3`: Move backward.
    The observation space consists of a dictionary:
    - `scent`: Vector with shape `[S]`, where `S` is the
      scent dimensionality.
    - `vision`: Matrix with shape `[2R+1, 2R+1, V]`,
      where `R` is the vision range and `V` is the
      vision/color dimensionality.
    - `moved`: Binary value indicating whether the last
      action resulted in the agent moving.
    """

    def __init__(self, sim_config = make_config(), task: int = 0, render_mode=None, f_type="obj"):
        """
        Creates a new JBW environment for OpenAI gym.
        Arguments:
        sim_config(SimulatorConfig) Simulator configuration
                                                                  to use.
        render_mode(bool)                Boolean value indicating
                                                                  whether or not to support
                                                                  rendering the
                                                                  environment.
        """
        self.sim_config = sim_config
        self._sim = None
        self._painter = None
        self._render = bool(render_mode)
        self.render_mode = render_mode
        self.T = 0
        self.f_type = f_type

        self.interval = 200_000
        self.half_interval = self.interval // 2
        self.quarter_interval = self.interval // 4

        # Computing shapes for the observation space.
        self.hash_dict = {(0, 0, 0): 0, (1, 0, 0): 1, (0, 1, 0): 2, (0, 0, 1): 3}
        self.shape_set = [(2, 2), (3, 3), (4, 4)]
        self.get_features = self.feature_picker(self.f_type)

        self.reset()

        scent_dim = len(self.sim_config.items[0].scent)
        vision_dim = len(self.sim_config.items[0].color)
        vision_range = self.sim_config.vision_range
        vision_shape = [2 * vision_range + 1, 2 * vision_range + 1, vision_dim]

        min_float = np.finfo(np.float32).min
        max_float = np.finfo(np.float32).max
        min_scent = min_float * np.ones(scent_dim)
        max_scent = max_float * np.ones(scent_dim)
        min_vision = min_float * np.ones(vision_shape)
        max_vision = max_float * np.ones(vision_shape)

        # Observations in this environment consist of a scent
        # vector, a vision matrix, and a binary value
        # indicating whether the last action resulted in the
        # agent moving.
        self.observation_space = spaces.Box(low=np.zeros(self.t_size), high=np.ones(self.t_size))
        self.scent_space = spaces.Box(low=min_scent, high=max_scent)
        self.action_space = spaces.Discrete(4)
        self.feature_space = spaces.Box(low=np.zeros(self.t_size), high=np.ones(self.t_size))

    def convert(self, vector):
        vector = np.ceil(vector)
        tuple_vec = tuple(vector)
        channel = self.hash_dict[tuple_vec]
        return channel

    def feature_picker(self, ftype="hash"):
        if ftype == "obj":
            self.t_size = (len(self.hash_dict) - 1) * (self.sim_config.vision_range * 2 + 1) ** 2

            def feature_func(vision_state):
                features = []
                obs_channel = np.apply_along_axis(self.convert, 2, vision_state)
                obs_channel = obs_channel.flatten()
                features = np.zeros((obs_channel.size, len(self.hash_dict)))
                features[np.arange(obs_channel.size), obs_channel] = 1
                return features[:, 1:].flatten()

            return feature_func

    def reward_fn(self, prev_items, items):
        timestep_interval = self.T % self.half_interval
        jellybean_weight = 1 - timestep_interval / self.quarter_interval
        if (self.T // self.half_interval) % 2 == 1:
            jellybean_weight *= -1
        onion_weight = -1 * jellybean_weight
        delta_items = items - prev_items
        reward = onion_weight * delta_items[0] + 0.1 * delta_items[1] + jellybean_weight * delta_items[2]
        return reward.astype(np.float32)

    def step(self, action):
        prev_items = self._agent.collected_items()

        self._agent._next_action = action
        self._agent.do_next_action()

        items = self._agent.collected_items()
        reward = self.reward_fn(prev_items, items)
        done = False

        self.vision_state = self._agent.vision()
        self.scent_state = self._agent.scent()
        self.T += 1

        return self.vision_state, reward, done, done, {}

    def reset(self, seed=None, options=None):
        """Resets this environment to its initial state."""
        del self._sim
        gc.collect()
        self.sim_config.seed = seed if seed is not None else 0
        self._sim = Simulator(sim_config=self.sim_config)
        self._agent = _JBWEnvAgent(self._sim)
        self.T = 0
        self.hash_vals = np.random.randint(
            0,
            np.iinfo(np.int32).max,
            size=(
                1 + len(self.sim_config.items),
                (2 * self.sim_config.vision_range) + 1,
                (2 * self.sim_config.vision_range) + 1,
            ),
        )
        self.hash_vals[0, :, :] = 0
        if self._render:
            del self._painter
            self._painter = MapVisualizer(self._sim, self.sim_config, bottom_left=(-70, -70), top_right=(70, 70))
        self.vision_state = self._agent.vision()
        self.scent_state = self._agent.scent()
        return self.vision_state, {}

    def render(self, mode="rgb_array"):
        if self.render_mode == "matplotlib":
            self._painter.draw()
        if self.render_mode == "rgb_array":
            self._painter.draw()
            canvas = self._painter._fig.canvas
            return np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(*reversed(canvas.get_width_height()), 3)
        elif not self._render:
            logger.warn("Need to pass `render=True` to support rendering.")
        else:
            logger.warn('Invalid rendering mode "%s". Only "matplotlib" is supported.')

    def close(self):
        del self._sim


class _JBWEnvAgent(Agent):
    """
    Helper class for the JBW environment, that represents
    a JBW agent living in the simulator.
    """

    def __init__(self, simulator):
        """
        Creates a new JBW environment agent.
        Arguments: simulator(Simulator)  The simulator the agent lives in.
        """
        super(_JBWEnvAgent, self).__init__(simulator, load_filepath=None)
        self._next_action = None

    def do_next_action(self):
        if self._next_action == 0:
            self.move(RelativeDirection.FORWARD)
        elif self._next_action == 1:
            self.move(RelativeDirection.LEFT)
        elif self._next_action == 2:
            self.move(RelativeDirection.RIGHT)
        elif self._next_action == 3:
            self.move(RelativeDirection.BACKWARD)
        else:
            logger.warn("Ignoring invalid action %d." % self._next_action)

    def save(self, filepath):
        pass

    def _load(self, filepath):
        pass
