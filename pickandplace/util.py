# Copyright 2024 @With-Robot 3rd
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

import numpy as np

from dataclasses import dataclass
from enum import Enum


#
# An Enum class for defining the state of a robot
#
class State(Enum):
    StandBy = 0
    MoveToPick = 1
    FindTarget = 2
    ApproachToTarget = 3
    PickTarget = 4
    MoveToPlace = 5
    PlaceTarget = 6
    MoveToBase = 7


#
# A class defining the settings required for robot operation
#
@dataclass(frozen=True)
class Config:
    pass


#
# A class defining the mission for robot
#
@dataclass(frozen=True)
class Mission:
    pick_location: str
    place_location: str
    target: str


#
# A class defining the context for robot operation
#
@dataclass
class Context:
    mission: Mission = None
    state: State = State.StandBy
    state_counter: int = 0
    base: np.array = None

    def set_state(self, state):
        self.state = state
        self.state_counter = 0

    def inc_state_counter(self):
        self.state_counter += 1


#
# A class defining readable datas of robot
#
class ReadData:
    localization: np.array = None
    wheels: list = None
    joints: list = None
    scans: list = None
    img: np.array = None


#
# A class defining control datas of robot
#
class ControlData:
    wheels_velocity: list = None
    wheels_position: list = None
    joints_position: list = None
    delta: int = None
