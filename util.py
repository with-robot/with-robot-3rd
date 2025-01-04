# Copyright 2024 @With-Robot 3.5
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
    map_size: tuple = (100, 100)
    map_cell: float = 0.1
    lidar_offset: float = 0.2751  # distance from youBot_ref to lidar
    lidar_pcd: int = 342 * 2  # Point Cloud Density
    place = {
        "/bedroom1": (40, 15),
        "/bedroom2": (75, 10),
        "/toilet": (85, 30),
        "/enterance": (80, 50),
        "/dining": (91, 80),
        "/lvingroom": (30, 85),
        "/balcony_init": (5, 65),
        "/balcony_end": (5, 20),
    }


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
    map: np.array = None
    map_loc: np.array = None
    mission: Mission = None
    state: State = State.StandBy
    state_count: int = 0

    base: tuple = None
    path: list = None
    path_idx: int = None

    mainpulator_state: int = 0
    manipulator_control_target: tuple = None  # manipulator control target

    target_index: int = -1

    def set_state(self, state):
        self.state = state
        self.state_count = 0

    def inc_state_counte(self):
        self.state_count += 1


#
# A class defining readable datas of robot
#
@dataclass
class ReadData:
    localization: np.array = None
    joints: np.array = None
    scan_flg: bool = False
    scan: np.array = None
    scan_position: tuple = None
    img_flag: bool = False
    img: np.array = None

    cam_localization: np.array = None


#
# A class defining control datas of robot
#
@dataclass
class ControlData:
    wheels_position: tuple = (
        np.deg2rad(0),
        np.deg2rad(0),
        np.deg2rad(0),
        np.deg2rad(0),
    )
    manipulator_position: tuple = (
        np.deg2rad(0),
        np.deg2rad(45),
        np.deg2rad(-120),
        np.deg2rad(-60),
        np.deg2rad(0),
    )
    gripper: bool = False
