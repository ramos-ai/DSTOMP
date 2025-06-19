import os
from datetime import datetime
from enum import StrEnum
from os.path import join
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from environment.gridworld import State
from environment.room_design import (
    larger_hallway_larger_room,
    larger_room,
    stomp_four_room_design,
    stomp_two_room_design,
)


class ExperimentsAvailable(StrEnum):
    TWO_ROOM_WITH_SUCCESSOR = "two_room_with_successor_representation"
    FOUR_ROOM_WITH_SUCCESSOR = "four_room_with_successor_representation"
    LARGER_ROOM_WITH_SUCCESSOR = "larger_room_with_successor_representation"
    LARGER_HALLWAY_LARGER_ROOM_WITH_SUCCESSOR = (
        "larger_hallway_larger_room_with_successor_representation"
    )


class Experiment(BaseModel):
    name: str
    env_design: List[List[int]]
    alpha_step_size: float
    num_subgoals_for_successor_representation: int
    off_policy_steps_for_successor_representation: int
    off_policy_steps_for_stomp_progression: int
    num_lookahead_operations: int
    hallways_states_info: Optional[Dict[int, State]] = None


__two_room_with_successor = Experiment(
    name="two_room_with_successor_representation",
    env_design=stomp_two_room_design,
    alpha_step_size=1.0,
    num_subgoals_for_successor_representation=4,
    off_policy_steps_for_successor_representation=int(10e6),
    off_policy_steps_for_stomp_progression=50_000,
    num_lookahead_operations=6_000,
    hallways_states_info={30: (7, 3)},
)

__four_room_with_successor = Experiment(
    name="four_room_with_successor_representation",
    env_design=stomp_four_room_design,
    alpha_step_size=0.05,
    num_subgoals_for_successor_representation=5,
    off_policy_steps_for_successor_representation=int(10e6),
    off_policy_steps_for_stomp_progression=500_000,
    num_lookahead_operations=20_000,
    hallways_states_info={51: (2, 6), 87: (6, 10), 62: (9, 7), 25: (6, 3)},
)

__larger_room_with_successor = Experiment(
    name="larger_room_with_successor_representation",
    env_design=larger_room,
    alpha_step_size=0.05,
    num_subgoals_for_successor_representation=10,
    off_policy_steps_for_successor_representation=int(10e6),
    off_policy_steps_for_stomp_progression=500_000,
    num_lookahead_operations=20_000,
    hallways_states_info={
        56: (5, 3),
        67: (16, 3),
        45: (23, 2),
        49: (27, 2),
        99: (25, 4),
        147: (3, 7),
        148: (13, 7),
        156: (27, 7),
        235: (16, 11),
        240: (25, 11),
        277: (14, 13),
        288: (27, 13),
        363: (26, 16),
        432: (25, 19),
        477: (18, 21),
        455: (23, 20),
        554: (18, 24),
        535: (27, 23),
        612: (27, 26),
    },
)

__larger_hallway_larger_room_with_successor = Experiment(
    name="larger_hallway_larger_room_with_successor_representation",
    env_design=larger_hallway_larger_room,
    alpha_step_size=0.05,
    num_subgoals_for_successor_representation=10,
    off_policy_steps_for_successor_representation=int(10e6),
    off_policy_steps_for_stomp_progression=500_000,
    num_lookahead_operations=20_000,
    hallways_states_info={
        481: (15, 22),
        479: (8, 22),
        483: (20, 22),
        484: (22, 22),
        486: (25, 22),
        111: (26, 5),
        83: (14, 4),
        41: (15, 22),
        28: (2, 2),
        381: (2, 18),
        119: (6, 6),
        436: (10, 20)
    },
)


def get_experiment(name: ExperimentsAvailable) -> Tuple[Experiment, str]:
    """
    Get the experiment configuration by name.
    """
    experiment_folder_path = join(
        "results", name, f"{datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}"
    )
    os.makedirs(name=experiment_folder_path, exist_ok=True)
    if name == __two_room_with_successor.name:
        return __two_room_with_successor, experiment_folder_path
    elif name == __four_room_with_successor.name:
        return __four_room_with_successor, experiment_folder_path
    elif name == __larger_room_with_successor.name:
        return __larger_room_with_successor, experiment_folder_path
    elif name == __larger_hallway_larger_room_with_successor.name:
        return __larger_hallway_larger_room_with_successor, experiment_folder_path
    else:
        raise ValueError(f"Experiment '{name}' not found.")
