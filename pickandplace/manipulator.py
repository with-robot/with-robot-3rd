# Copyright 2024 @With-Robot 3rd
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import cv2

from util import State, Config, Context, ReadData, ControlData


# pi / 2
PI_HALF = np.pi / 2

config: Config = Config()
visual_objects = [None] * 128


# find target
def find_target(context: Context, youbot_data: ReadData):
    result: bool = False
    control_data: ControlData = ControlData()

    # move joint to init position
    if context.state_counter == 1:
        context.mainpulator_state = 0
        context.target_location = None
        control_data.joints_position = config.find_target_init.copy()
        control_data.camera_call_back = find_target_cb
    else:
        result = True if context.target_location is not None else False

    return result, control_data


# callback for find target
def find_target_cb(context: Context, youbot_data: ReadData, control_data: ControlData):
    def diff():
        return sum(
            [
                abs(youbot_data.joints[i] - control_data.joints_position[i])
                for i in range(len(config.find_target_init))
            ]
        )

    state = context.mainpulator_state
    if state == 0:  # move init position
        if diff() < 0.005:
            context.mainpulator_state = 1
        return

    visualize(youbot_data.img)

    if context.target_location is None:
        bboxs = detect_red_box(youbot_data.img)
        if bboxs:
            near_index = find_nearest_bbox(youbot_data.img, bboxs)
            context.target_location = bboxs[near_index]

        if 1 <= state <= 5:  # find target
            control_data.joints_position = config.find_target_init.copy()
            control_data.joints_position[0] = (
                np.pi / 2 if state % 2 == 1 else -np.pi / 2
            )
            control_data.joints_position[1] += config.find_target_delta_up * (state - 1)
            if diff() < 0.005:
                context.mainpulator_state += 1
        else:  # fail to find target, move to base
            context.set_state(State.MoveToBase)
    else:  # move camera to center
        bboxs = detect_red_box(youbot_data.img)
        if bboxs:
            near_index = find_nearest_bbox(youbot_data.img, bboxs)
            context.target_location = bboxs[near_index]

            base_x = youbot_data.img.shape[1] // 2
            base_y = youbot_data.img.shape[0] // 2
            x, y, w, h = context.target_location
            center_x = x + w // 2
            center_y = y + h // 2
            diff_x = center_x - base_x
            diff_y = center_y - base_y

            control_data.joints_position = youbot_data.joints[:5]
            control_data.joints_position[0] -= diff_x / 1000
            control_data.joints_position[1] -= diff_y / 1000

            if diff() < 0.005:  # calc target location
                _, pc_hat = fk(youbot_data.joints[1:], [youbot_data.joints[0]])
                xyz = R.from_quat(pc_hat[3:]).as_euler("xyz")
                theta_z = np.pi + xyz[0]
                theta_x = xyz[2]
                dist = pc_hat[2] * np.tan(theta_z)
                pt_hat = pc_hat[:3] + dist * np.array(
                    [-np.sin(theta_x), np.cos(theta_x), 0]
                )
                pt_hat[-1] = 0.02
                context.target_location = pt_hat
                return True


# pick target
def pick_target(context: Context, youbot_data: ReadData):
    result: bool = False
    control_data: ControlData = ControlData()

    if context.state_counter == 1:
        context.mainpulator_state = 0
    context.mainpulator_state += 1

    visualize(youbot_data.img)

    #### this section will be changed by path planning ####
    # move joint to above of target position
    if context.mainpulator_state == 1:
        pt_hat = context.target_location.copy()
        pt_hat[-1] = 0.1
        target_thetas = solve(youbot_data.joints, pt_hat)
        control_data.joints_position = target_thetas
    #### this section will be changed by visual servoing ####
    # move joint to target position
    elif context.mainpulator_state == 2:
        pt_hat = context.target_location.copy()
        target_thetas = solve(youbot_data.joints, pt_hat)
        control_data.joints_position = target_thetas
        print(target_thetas)
    #########################################################
    # grip target
    elif context.mainpulator_state == 3:
        control_data.gripper_state = True
    #### this section will be changed by path planning ####
    # move joint to loading location
    elif context.mainpulator_state == 4:
        control_data.joints_position = config.loading_location.copy()
        control_data.joints_position[0] = 0
        control_data.joints_position[1] = 0
    # move joint to loading location
    elif context.mainpulator_state == 5:
        control_data.joints_position = config.loading_location.copy()
        control_data.joints_position[1] = 0
    # move joint to loading location
    elif context.mainpulator_state == 6:
        control_data.joints_position = config.loading_location.copy()
    #########################################################
    # drop target
    elif context.mainpulator_state == 7:
        control_data.gripper_state = False
    # move joint to above of loaded target
    elif context.mainpulator_state == 8:
        control_data.joints_position = config.loading_location.copy()
        control_data.joints_position[1] += 0.2
        control_data.joints_position[2] -= 0.1
    # check loaded target
    elif context.mainpulator_state == 9:
        bboxs = detect_red_box(youbot_data.img)
        if len(bboxs) > 0:
            result = True
        else:  # re try
            context.mainpulator_state = 0

    return result, control_data


# place target
def place_target(context: Context, youbot_data: ReadData):
    result: bool = False
    control_data: ControlData = ControlData()

    if context.state_counter == 1:
        context.mainpulator_state = 0
    context.mainpulator_state += 1

    visualize(youbot_data.img)

    # move joint to loading location
    if context.mainpulator_state == 1:
        control_data.joints_position = config.loading_location.copy()
    # grip target
    elif context.mainpulator_state == 2:
        control_data.gripper_state = True
    # move joint to place location
    elif context.mainpulator_state == 3:
        control_data.joints_position = config.loading_location.copy()
        control_data.joints_position[1] = 0
    # move joint to place location
    elif context.mainpulator_state == 4:
        control_data.joints_position = config.loading_location.copy()
        control_data.joints_position[0] = 0
        control_data.joints_position[1] = 0
    # move joint to place location
    elif context.mainpulator_state == 5:
        control_data.joints_position = config.place_location.copy()
    # drop target
    elif context.mainpulator_state == 6:
        control_data.gripper_state = False
    # move to base postion
    elif context.mainpulator_state == 7:
        control_data.joints_position = config.base_location.copy()
        result = True

    return result, control_data


def detect_red_box(img):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # red color HSV range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    # make red mask
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    # find coutoures
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # bboxs
    bboxs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxs.append((x, y, w, h))
    return bboxs


# find nearest bbox
def find_nearest_bbox(img, bboxs):
    base_x = img.shape[1] // 2
    zero_y = img.shape[0]

    near_zero_d = 0xFFFFFFFF
    near_index = -1
    for i, (x, y, w, h) in enumerate(bboxs):
        center_x = x + w // 2
        center_y = y + h // 2
        zero_d = (base_x - center_x) ** 2 + (zero_y - center_y) ** 2
        if zero_d < near_zero_d:
            near_zero_d = zero_d
            near_index = i
    return near_index


# visualize camera image
def visualize(img):
    # remove visual objects
    for i in range(len(visual_objects)):
        if visual_objects[i] is None:
            break
        visual_objects[i].remove()
        visual_objects[i] = None
    # display image
    visual_objects[0] = plt.imshow(img)
    plt.pause(0.001)


# forward kinematics
def fk(thetas, params):
    j1, j2, j3, j4 = thetas[:4]
    j0 = params[0]

    # 자동차 -> joint-0
    TC0 = np.array(
        [  # 좌표이동 및 y축을 기준으로 90도 회전
            [1, 0, 0, 0.0],
            [0, 1, 0, 0.166],
            [0, 0, 1, 0.099],
            [0, 0, 0, 1],
        ]
    ) @ np.array(
        [
            [np.cos(j0), -np.sin(j0), 0, 0],
            [np.sin(j0), np.cos(j0), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # joint-0 -> joint-1
    ay1 = PI_HALF
    T01 = np.array(
        [  # 좌표이동 및 y축을 기준으로 90도 회전
            [np.cos(ay1), 0, np.sin(ay1), 0.0],
            [0, 1, 0, 0.033],
            [-np.sin(ay1), 0, np.cos(ay1), 0.147],
            [0, 0, 0, 1],
        ]
    ) @ np.array(
        [  # z축을 기준으로 j1만큼 회전
            [np.cos(j1), -np.sin(j1), 0, 0],
            [np.sin(j1), np.cos(j1), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    TC1 = TC0 @ T01

    # joint-1 -> joint-2
    T12 = np.array(
        [  # 좌표이동, 회전 없음
            [1, 0, 0, -0.155],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1],
        ]
    ) @ np.array(
        [  # z축을 기준으로 j2만큼 회전
            [np.cos(j2), -np.sin(j2), 0, 0],
            [np.sin(j2), np.cos(j2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    TC2 = TC1 @ T12

    # joint-2 -> joint-3
    T23 = np.array(
        [  # 좌표이동, 회전 없음
            [1, 0, 0, -0.135],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1],
        ]
    ) @ np.array(
        [  # z축을 기준으로 j3만큼 회전
            [np.cos(j3), -np.sin(j3), 0, 0],
            [np.sin(j3), np.cos(j3), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    TC3 = TC2 @ T23

    # joint-3 -> joint-4
    ay4 = -PI_HALF
    T34 = np.array(
        [  # 좌표이동 및 y축을 기준으로 -90도 회전
            [np.cos(ay4), 0, np.sin(ay4), -0.081],
            [0, 1, 0, 0.0],
            [-np.sin(ay4), 0, np.cos(ay4), 0.0],
            [0, 0, 0, 1],
        ]
    ) @ np.array(
        [  # z축을 기준으로 j4만큼 회전
            [np.cos(j4), -np.sin(j4), 0, 0],
            [np.sin(j4), np.cos(j4), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    TC4 = TC3 @ T34

    pe_hat = TC4 @ np.array([0.0, 0.0, 0.123, 1])
    pc_hat = TC4 @ np.array([0.0, 0.0, 0.075, 1])
    oc_hat = R.from_matrix(TC4[:-1, :-1]).as_quat()

    return pe_hat[:3], np.concatenate((pc_hat[:3], oc_hat))


def ik(thetas, params):
    pt = params[-1][:3]
    pe_hat, _ = fk(thetas, params)
    # theta 범위 검증
    if thetas[0] < np.deg2rad(-90) or np.deg2rad(75) < thetas[0]:
        return 10, 0, 0, 0
    elif thetas[1] < np.deg2rad(-131.00) or np.deg2rad(131.00) < thetas[1]:
        return 10, 0, 0, 0
    elif thetas[2] < np.deg2rad(-102.00) or np.deg2rad(102.00) < thetas[2]:
        return 10, 0, 0, 0
    elif thetas[3] < np.deg2rad(-90.00) or np.deg2rad(90.00) < thetas[3]:
        return 10, 0, 0, 0
    return np.linalg.norm(pe_hat - pt), 0, 0, 0


def solve(js, pt):
    target_thetas = fsolve(ik, [js[1], js[2], js[3], js[4]], [js[0], pt])
    target_thetas[3] = 0
    return np.concatenate((np.array([js[0]]), target_thetas))
