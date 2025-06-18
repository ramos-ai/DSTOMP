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
)


class ExperimentsAvailable(StrEnum):
    FOUR_ROOM_WITH_SUCCESSOR = "four_room_with_successor_representation"
    LARGER_ROOM_WITH_SUCCESSOR = "larger_room_with_successor_representation"
    LARGER_HALLWAY_LARGER_ROOM_WITH_SUCCESSOR = (
        "larger_hallway_larger_room_with_successor_representation"
    )


class Experiment(BaseModel):
    name: str
    env_design: List[List[int]]
    num_subgoals: int
    off_policy_steps_for_successor_representation: int
    off_policy_steps_for_stomp_progression: int
    num_lookahead_operations: int
    hallways_states_info: Optional[Dict[int, State]] = None


__four_room_with_successor = Experiment(
    name="four_room_with_successor_representation",
    env_design=stomp_four_room_design,
    num_subgoals=5,
    off_policy_steps_for_successor_representation=int(10e6),
    off_policy_steps_for_stomp_progression=500_000,
    num_lookahead_operations=20_000,
    hallways_states_info={51: (2, 6), 87: (6, 10), 62: (9, 7), 25: (6, 3)},
)

__larger_room_with_successor = Experiment(
    name="larger_room_with_successor_representation",
    env_design=larger_room,
    num_subgoals=10,
    off_policy_steps_for_successor_representation=int(10e6),
    off_policy_steps_for_stomp_progression=500_000,
    num_lookahead_operations=20_000,
)

__larger_hallway_larger_room_with_successor = Experiment(
    name="larger_hallway_larger_room_with_successor_representation",
    env_design=larger_hallway_larger_room,
    num_subgoals=10,
    off_policy_steps_for_successor_representation=int(10e6),
    off_policy_steps_for_stomp_progression=500_000,
    num_lookahead_operations=20_000,
)


def get_experiment(name: ExperimentsAvailable) -> Tuple[Experiment, str]:
    """
    Get the experiment configuration by name.
    """
    experiment_folder_path = join(
        "results", name, f"{datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}"
    )
    os.makedirs(name=experiment_folder_path, exist_ok=True)
    if name == __four_room_with_successor.name:
        return __four_room_with_successor, experiment_folder_path
    elif name == __larger_room_with_successor.name:
        return __larger_room_with_successor, experiment_folder_path
    elif name == __larger_hallway_larger_room_with_successor.name:
        return __larger_hallway_larger_room_with_successor, experiment_folder_path
    else:
        raise ValueError(f"Experiment '{name}' not found.")
