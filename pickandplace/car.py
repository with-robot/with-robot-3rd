# Copyright 2024 @With-Robot 3rd
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

from util import Context, ReadData, ControlData


# get current location of robot
def get_location(context: Context, youbot_data: ReadData):
    location = None
    return location


# move car to pick room
def move_to_pick(context: Context, youbot_data: ReadData):
    result: bool = True
    control_data: ControlData = ControlData()

    return result, control_data


# move car near to target location
def approach_to_target(context: Context, youbot_data: ReadData):
    result: bool = True
    control_data: ControlData = ControlData()

    return result, control_data


# move car to place room
def move_to_place(context: Context, youbot_data: ReadData):
    result: bool = True
    control_data: ControlData = ControlData()

    return result, control_data


# move car to base location
def move_to_base(context: Context, youbot_data: ReadData):
    result: bool = True
    control_data: ControlData = ControlData()

    return result, control_data
