# Copyright 2024 @With-Robot 3rd
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

from util import Context, Config, ConfigObj, ReadData, ControlData
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import numpy as np

sim = RemoteAPIClient().require("sim")
simOMPL = RemoteAPIClient().require("simOMPL")

config: Config = Config()
configobj: ConfigObj = ConfigObj()

robotHandle = configobj.youBot
refHandle = configobj.youBot_ref
collVolumeHandle = configobj.youBot_collision_box
wheels = configobj.wheels
predefined_points = configobj.predefined_points


# get current location of robot
def get_location(context: Context, youbot_data: ReadData):
    # youbot 의 현재 위치
    location = youbot_data.localization
    pick_location = str(input("Input the pick location : "))
    place_location = str(input("Input the place location : "))
    pick_location_id: int = None
    place_location_id: int = None
    try:
        # Ensure pick and place locations are different
        if pick_location_id == place_location_id:
            raise ValueError("Pick and place locations must be different.")
        # Get IDs for pick and place locations
        if pick_location in predefined_points:
            pick_location_id = predefined_points[pick_location]
        else:
            raise ValueError(f"Invalid pick location: {pick_location}")
        if place_location in predefined_points:
            place_location_id = predefined_points[place_location]
        else:
            raise ValueError(f"Invalid place location: {place_location}")

        # base location (position info.)
        context.base_goal_location = location
        # pick_location (dummy id)
        context.pick_location_id = pick_location_id
        # place location (dummy id)
        context.place_location_id = place_location_id

    except ValueError as e:
        print(e)

    # location : 현재 위치
    return location


# move car to pick room
def move_to_pick(context: Context, youbot_data: ReadData):
    result: bool = True
    control_data: ControlData = ControlData()
    # main 에서 context.pick_goal_location 에 configobj.predefined_points 의 어느 장소를 입력해야하는 내용 추가해야함.
    if context.pick_location_id is not None:
        dummyPos = sim.getObjectPosition(context.pick_location_id)
        control_data.control_cb = move_cb(goal=dummyPos)

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
    # main 에서 context.place_goal_location 에 configobj.predefined_points 의 어느 장소를 입력해야하는 내용 추가해야함.
    if context.place_location_id is not None:
        dummyPos = sim.getObjectPosition(context.place_location_id)
        control_data.control_cb = move_cb(goal=dummyPos)

    return result, control_data


# move car to base location
def move_to_base(context: Context, youbot_data: ReadData):
    result: bool = True
    control_data: ControlData = ControlData()
    # main 에서 context.base_goal_location 에 configobj.predefined_points 의 어느 장소를 입력해야하는 내용 추가해야함.
    if context.base_goal_location is not None:
        control_data.control_cb = move_cb(goal=context.base_goal_location)

    return result, control_data


# callback 안에서 전체 주행이 완료되어야 함
# mecanumwheel control 을 main 에서가 아닌 여기서 실행을 시킴.
def move_cb(context: Context, youbot_data: ReadData, control_data: ControlData, goal):
    obstaclesCollection = sim.createCollection(0)
    sim.addItemToCollection(obstaclesCollection, sim.handle_all, -1, 0)
    sim.addItemToCollection(obstaclesCollection, sim.handle_tree, robotHandle, 1)
    collPairs = [collVolumeHandle, obstaclesCollection]

    def findPath(goalPos):
        search_algo = simOMPL.Algorithm.BiTRRT
        while not control_data.path_flag:
            task = simOMPL.createTask("t")
            simOMPL.setAlgorithm(task, search_algo)
            # 원래 여기에 sim.getObjectPosition(refHandle, -1)
            startPos = youbot_data.localization[:3]  # [x,y,z,qw,qx,qy,qz]
            ss = [
                simOMPL.createStateSpace(
                    "2d",
                    simOMPL.StateSpaceType.position2d,
                    collVolumeHandle,
                    [
                        startPos[0] - config.search_range,
                        startPos[1] - config.search_range,
                    ],
                    [
                        startPos[0] + config.search_range,
                        startPos[1] + config.search_range,
                    ],
                    1,
                )
            ]
            simOMPL.setStateSpace(task, ss)
            simOMPL.setCollisionPairs(task, collPairs)
            simOMPL.setStartState(task, startPos[:2])
            simOMPL.setGoalState(task, goalPos[:2])
            simOMPL.setStateValidityCheckingResolution(task, 0.01)
            simOMPL.setup(task)

            if simOMPL.solve(task, config.search_duration):
                simOMPL.simplifyPath(task, config.search_duration)
                path = simOMPL.getPath(task)

                visualizePath(path)

                return path
            else:
                # path planning 실패 시 예외처리해야함.
                pass
            time.sleep(0.01)

    def followPath(goalPos, path: list = None):
        if path:
            path_3d = []
            for i in range(0, len(path) // 2):
                path_3d.extend([path[2 * i], path[2 * i + 1], 0.0])
            prev_dist = 0
            track_pos_container = sim.addDrawingObject(
                sim.drawing_spherepoints | sim.drawing_cyclic,
                0.02,
                0,
                -1,
                1,
                [1, 0, 1],
            )
            while True:
                # youbot_data 는 실시간 처리?
                currPos = youbot_data.localization[:3]
                pathLength, totalDist = sim.getPathLenghts(path_3d, 3)
                closet_dist = sim.getClosestPosOnPath(path_3d, pathLength, currPos)

                if closet_dist <= prev_dist:
                    closet_dist += totalDist / 200
                prev_dist = closet_dist

                # refine the path smoothly
                targetPoint = sim.getPathInterpolateConfig(
                    path_3d, pathLength, closet_dist
                )
                sim.addDrawingObjectItem(track_pos_container, targetPoint)

                # calculating the velocity for each of the 4 mecanum wheels
                m = youbot_data.robot_mat
                m_inv = sim.getMatrixInverse(m)
                rel_p = sim.multiplyVector(m_inv, targetPoint)
                # control yaw
                rel_o = math.atan2(rel_p[1], rel_p[0]) - math.pi / 2

                p_parm = config.drive_parms["p_parm"]
                p_parm_rot = config.drive_parms["p_parm_rot"]
                max_v = config.drive_parms["max_v"]
                max_v_rot = config.drive_parms["max_v_rot"]
                accel_f = config.drive_parms["accel_f"]

                forwback_vel = rel_p[1] * p_parm
                side_vel = rel_p[0] * p_parm
                v = (forwback_vel**2 + side_vel**2) ** 0.5
                if v > max_v:
                    forwback_vel *= max_v / v
                    side_vel *= max_v / v
                """
                Since the manipulator's direction is assumed to be the front, 
                the opposite direction of the original body front is set as the forward-front direction, 
                using a negative sign.
                """
                rot_vel = -rel_o * p_parm_rot
                if abs(rot_vel) > max_v_rot:
                    rot_vel = max_v_rot * rot_vel / abs(rot_vel)
                # 여기도 wheels_velocity 가 업데이트 되도록 해야함.
                # 근데 wheels_position 은 main.py 에서 업데이트됨.
                prev_forwback_vel = control_data.wheels_velocity[0]
                prev_side_vel = control_data.wheels_velocity[1]
                prev_rot_vel = control_data.wheels_velocity[2]

                df = forwback_vel - prev_forwback_vel
                ds = side_vel - prev_side_vel
                dr = rot_vel - prev_rot_vel

                if abs(df) > max_v * accel_f:
                    df = max_v * accel_f * df / abs(df)
                if abs(ds) > max_v * accel_f:
                    ds = max_v * accel_f * ds / abs(ds)
                if abs(dr) > max_v_rot * accel_f:
                    dr = max_v_rot * accel_f * dr / abs(dr)

                forwback_vel = prev_forwback_vel + dr
                side_vel = prev_side_vel + ds
                rot_vel = prev_rot_vel + dr

                # control_data 에 저장
                control_data.wheels_velocity[0] = forwback_vel
                control_data.wheels_velocity[1] = side_vel
                control_data.wheels_velocity[2] = rot_vel

                # set joint 부분은 main 에서 처리. -> x, move_to_pick_cb 루프 안에서 완료
                mecanum_wheel_control()

                if (
                    np.linalg.norm(
                        np.array(sim.getObjectPosition(goalPos, -1))
                        - np.array(sim.getObjectPosition(refHandle, -1))
                    )
                    < 0.6
                ):

                    sim.removeDrawingObject(track_pos_container)
                    break

    goalPos = goal
    path = findPath(goalPos=goalPos)
    if path is not None:
        followPath(goalPos=goalPos, path=path)
        # setting the velocity to zero for stopping
        for i, vel in enumerate(config.drive_stop_parms):
            control_data.wheels_velocity[i] = vel
        mecanum_wheel_control()
        clear_path()
    else:
        print("Path is not found.")


def visualizePath(path):
    control_data: ControlData = ControlData()
    if control_data.line_container is None:
        control_data.line_container = sim.addDrawingObject(
            sim.drawing_lines, 3, 0, -1, 99999, [0.2, 0.2, 0.2]
        )
    sim.addDrawingObjectItem(control_data.line_container, None)

    if path:
        for i in range(1, len(path) // 2):
            line_data = [
                path[2 * i - 2],
                path[2 * i - 1],
                0.001,
                path[2 * i],
                path[2 * i + 1],
                0.001,
            ]
            sim.addDrawingObjectItem(control_data.line_container, line_data)


# remove line_container elements to re-draw the line
def clear_path():
    control_data: ControlData = ControlData()
    sim.removeDrawingObject(control_data.line_container)
    control_data.line_container = None


# mecanum wheel control
def mecanum_wheel_control(control_data: ControlData, youbot_data: ReadData):
    forwback_vel = control_data.wheels_velocity[0]
    side_vel = control_data.wheels_velocity[1]
    rot_vel = control_data.wheels_velocity[2]

    sim.setJointTargetVelocty(wheels[0], -forwback_vel - side_vel - rot_vel)
    sim.setJointTargetVelocty(wheels[1], -forwback_vel + side_vel - rot_vel)
    sim.setJointTargetVelocty(wheels[2], -forwback_vel + side_vel + rot_vel)
    sim.setJointTargetVelocty(wheels[0], -forwback_vel - side_vel + rot_vel)
