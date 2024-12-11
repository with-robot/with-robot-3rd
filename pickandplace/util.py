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
    # initial position of find target
    find_target_init = [-np.pi / 2, -np.pi / 4, -np.pi / 3, -np.pi / 3, 0.0]
    # delta theta to up of find target
    find_target_delta_up = np.pi / 16
    # robot loading location
    loading_location = [np.pi, -np.pi / 6, -np.pi / 2.7, -np.pi / 3, 0]
    # robot place location
    place_location = [0, -np.pi * 0.4, -np.pi * 0.285, -np.pi * 0.245, 0.0]
    # initial position of robot
    base_location = [0 / 2, -np.pi / 4, -np.pi / 3, -np.pi / 3, 0.0]
    # target height before IBVS
    target_height = 0.06
    # pixel threshold for IBVS
    ibvs_threshold = 90

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

    mainpulator_state: int = 0
    target_location: np.array = None
    tmp_target_theta: np.array = None
    pixelPositions = []

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
    cam_localization : np.array = None
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
    delta: int = 0.025
    gripper_state: bool = None
    exec_count = 0
    control_cb = None
    read_lidar: bool = False
    read_camera: bool = False
