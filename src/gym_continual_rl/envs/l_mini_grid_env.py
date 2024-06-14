from __future__ import annotations

from typing import Any

from gymnasium.core import ObsType
from gymnasium.spaces import Discrete
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class LMiniGridEnv(MiniGridEnv):
    """
    ## Description

    Classic four room reinforcement learning environment. The agent must
    navigate in a maze composed of four rooms interconnected by 4 gaps in the
    walls. To obtain a reward, the agent must reach the green goal square. Both
    the agent and the goal square are randomly placed in any of the four rooms.

    ## Mission Space

    "reach the goal"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-FourRooms-v0`

    """

    def __init__(
        self,
        agent_pos=None,
        goal_pos=None,
        **kwargs,
    ):
        self._agent_default_pos = agent_pos
        self._agent_default_dir = 0
        self._goal_default_pos = [
            {
                (7, 1): 5,
                (9, 3): 0,
            },
            {
                (7, 1): 0,
                (9, 3): 5,
            },
        ]
        self.wall_positions = [(3, 4), (4, 4)]
        self.task = 0

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=15,
            height=11,
            max_steps=100,
            see_through_walls=False,
            agent_view_size=5,
            **kwargs,
        )
        self.action_space = Discrete(self.actions.forward + 1)

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for wall_pos in self.wall_positions:
            self.put_obj(Wall(), *wall_pos)

        for i in list(range(5, 15)):
            self.grid.vert_wall(i, 4, 7)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._agent_default_dir
        else:
            self.agent_pos = self.place_agent(top=(1, 9), size=(4, 1))

        for goal_default_pos in self._goal_default_pos[self.task]:
            goal = Goal()
            self.put_obj(goal, *goal_default_pos)
            goal.init_pos, goal.cur_pos = goal_default_pos

    def _reward(self) -> float:
        return self._goal_default_pos[self.task][tuple(self.agent_pos)]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        if options is not None and "task" in options:
            self.task = options["task"]
        return super().reset(seed=seed, options=options)
