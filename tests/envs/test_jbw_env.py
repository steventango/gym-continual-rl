from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import gym_continual_rl  # noqa: F401
from gym_continual_rl.envs.jbw_env import get_reward_0, get_reward_1


UP = 0
LEFT = 1
RIGHT = 2
DOWN = 3


def test_jbw_env_task_0():
    env = gym.make("gym_continual_rl/JBW-v0", task=0)
    _ = env.reset(seed=0)
    assert env.unwrapped._reward_fn == get_reward_0

    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(LEFT)
        assert reward == 0
        assert not terminated
        assert not truncated

    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(UP)
        assert reward == 0
        assert not terminated
        assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert reward == -1
    assert not terminated
    assert not truncated

    for _ in range(8):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 0
        assert not terminated
        assert not truncated

    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 2
        assert not terminated
        assert not truncated

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 0
        assert not terminated
        assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert reward == 0.1
    assert not terminated
    assert not truncated


def test_jbw_env_task_1():
    env = gym.make("gym_continual_rl/JBW-v0", task=1)
    _ = env.reset(seed=0)
    assert env.unwrapped._reward_fn == get_reward_1

    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(LEFT)
        assert reward == 0
        assert not terminated
        assert not truncated

    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(UP)
        assert reward == 0
        assert not terminated
        assert not truncated

    obs, reward, terminated, truncated, info = env.step(UP)
    assert reward == 2
    assert not terminated
    assert not truncated

    for _ in range(8):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 0
        assert not terminated
        assert not truncated

    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == -1
        assert not terminated
        assert not truncated

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(RIGHT)
        assert reward == 0
        assert not terminated
        assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert reward == 0
    assert not terminated
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(DOWN)
    assert reward == 0.1
    assert not terminated
    assert not truncated

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

def make_config(permutation):
    # specify the item types
    items = []
    interaction_fns = [[InteractionFunction.ZERO], [InteractionFunction.ZERO], [InteractionFunction.ZERO]]
    interaction_fns = [interaction_fns[i] for i in permutation]
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
            interaction_fns=interaction_fns,
        )
    )
    interaction_fns = [
        [InteractionFunction.ZERO],
        [InteractionFunction.PIECEWISE_BOX, 3, 10, 1, -2],
        [InteractionFunction.PIECEWISE_BOX, 25, 50, -50, -10],
    ]
    interaction_fns = [interaction_fns[i] for i in permutation]
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
            interaction_fns=interaction_fns,
        )
    )
    interaction_fns = [
        [InteractionFunction.ZERO],
        [InteractionFunction.PIECEWISE_BOX, 25, 50, -50, -10],
        [InteractionFunction.PIECEWISE_BOX, 3, 10, 1, -2],
    ]
    interaction_fns = [interaction_fns[i] for i in permutation]
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
            interaction_fns=interaction_fns,
        )
    )
    items = [items[i] for i in permutation]
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


def test_jbw_env_render():
    env = gym.make("gym_continual_rl/JBW-v0", task=0, render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    image = Image.fromarray(frame)
    IMAGES_PATH = Path(__file__).parent.parent.parent / "images"
    expected_image = Image.open(IMAGES_PATH / "jbw.png")
    assert np.array_equal(np.array(image), np.array(expected_image))
    env.close()



def test_jbw_env_render_large():
    import itertools
    permutations = list(itertools.permutations([0, 1, 2]))
    for j, permutation in enumerate(permutations):
        env = gym.make("gym_continual_rl/JBW-v0", task=0, render_mode="rgb_array", sim_config=make_config(permutation))
        env.reset(seed=0)
        env.unwrapped._painter = MapVisualizer(env.unwrapped._sim, env.unwrapped.sim_config, bottom_left=(-512, -512), top_right=(512, 512))

        for i in range(1, 8):
            side_length = i * 128
            for _ in range(side_length // 2):
                obs, reward, terminated, truncated, info = env.step(UP)

            for _ in range(side_length // 2):
                obs, reward, terminated, truncated, info = env.step(RIGHT)

            for _ in range(side_length):
                obs, reward, terminated, truncated, info = env.step(DOWN)

            for _ in range(side_length):
                obs, reward, terminated, truncated, info = env.step(LEFT)

            for _ in range(side_length):
                obs, reward, terminated, truncated, info = env.step(UP)

            for _ in range(side_length // 2):
                obs, reward, terminated, truncated, info = env.step(RIGHT)

            for _ in range(side_length // 2):
                obs, reward, terminated, truncated, info = env.step(DOWN)

        frame = env.render()

        image = Image.fromarray(frame)
        IMAGES_PATH = Path(__file__).parent.parent.parent / "images"
        image.save(IMAGES_PATH / f"jbw_{j}_{'-'.join(map(str, permutation))}.png")
        # expected_image = Image.open(IMAGES_PATH / "jbw.png")
        # assert np.array_equal(np.array(image), np.array(expected_image))
        env.close()
