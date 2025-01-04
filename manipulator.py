# Copyright 2024 @With-Robot 3.5
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import cv2

from util import Config, Context, ReadData, ControlData


PI_HALF = np.pi / 2


#
# A class for manipulator operation
#
class ManipulatorClass:
    def __init__(self):
        self.config = Config()

    #
    # 가장 가까은 BBox 선택
    #
    def _find_nearest_bbox(self, img, bboxs):
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
        return bboxs[near_index] if near_index >= 0 else None

    #
    # BBox 검출
    #
    def _detect_red_box(self, img):
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
        if bboxs:
            return self._find_nearest_bbox(img, bboxs)
        else:
            return None

    #
    # Joint 제어
    #
    def _control_joint(self, manipulator_control_target, read_data: ReadData, control_data: ControlData):
        diff_sum = 0
        manipulator_position = []
        for i in range(len(manipulator_control_target)):
            diff = read_data.joints[i + 4] - manipulator_control_target[i]
            diff_sum += np.abs(diff)
            manipulator_position.append(read_data.joints[i + 4] - np.clip(diff, -0.025, 0.025))
        if diff_sum > 0.01:
            control_data.manipulator_position = tuple(manipulator_position)
        else:
            return True

    #
    # Joint 제어
    #
    def _control_bbox_center(self, bbox, read_data: ReadData, control_data: ControlData):
        base_x = read_data.img.shape[1] // 2
        base_y = read_data.img.shape[0] // 2
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        diff_x = (center_x - base_x) / 1000
        diff_y = (center_y - base_y) / 1000
        diff_sum = np.abs(diff_x) + np.abs(diff_y)
        control_data.manipulator_position = (
            read_data.joints[4] - np.clip(diff_x, -0.025, 0.025),
            read_data.joints[5],
            read_data.joints[6],
            read_data.joints[7] - np.clip(diff_y, -0.025, 0.025),
        )
        if diff_sum < 0.005:
            return True

    #
    # 로봇팔을 이용하여 Target 탐색
    #
    def find_target(self, context: Context, read_data: ReadData, control_data: ControlData):
        if context.state_count == 1:
            read_data.scan_flg = False
            read_data.img_flag = True

            context.mainpulator_state = 0
        else:
            bbox = self._detect_red_box(read_data.img)
            if bbox:
                return self._control_bbox_center(bbox, read_data, control_data)
            elif context.mainpulator_state == 0:
                manipulator_control_target = (
                    np.deg2rad(-90),
                    np.deg2rad(45),
                    np.deg2rad(-125),
                    np.deg2rad(-60),
                )
                if self._control_joint(manipulator_control_target, read_data, control_data):
                    context.mainpulator_state += 1
            elif context.mainpulator_state == 1:
                manipulator_control_target = (
                    np.deg2rad(90),
                    np.deg2rad(45),
                    np.deg2rad(-125),
                    np.deg2rad(-60),
                )
                if self._control_joint(manipulator_control_target, read_data, control_data):
                    context.mainpulator_state += 1
            elif context.mainpulator_state == 2:
                manipulator_control_target = (
                    np.deg2rad(-90),
                    np.deg2rad(45),
                    np.deg2rad(-125),
                    np.deg2rad(-50),
                )
                if self._control_joint(manipulator_control_target, read_data, control_data):
                    context.mainpulator_state += 1
            elif context.mainpulator_state == 3:
                manipulator_control_target = (
                    np.deg2rad(90),
                    np.deg2rad(45),
                    np.deg2rad(-125),
                    np.deg2rad(-40),
                )
                if self._control_joint(manipulator_control_target, read_data, control_data):
                    context.mainpulator_state += 1
            elif context.mainpulator_state == 4:
                manipulator_control_target = (
                    np.deg2rad(90),
                    np.deg2rad(45),
                    np.deg2rad(-125),
                    np.deg2rad(-30),
                )
                if self._control_joint(manipulator_control_target, read_data, control_data):
                    context.mainpulator_state += 1

    #
    # 휠을 이용하여 Target에 접근
    #
    def approach_to_target(self, context: Context, read_data: ReadData, control_data: ControlData):
        if context.state_count == 1:
            read_data.scan_flg = True
            read_data.img_flag = True
        else:
            bbox = self._detect_red_box(read_data.img)
            if not bbox:
                return
            self._control_bbox_center(bbox, read_data, control_data)
            diff = read_data.joints[4]
            if np.abs(diff) > 0.05:
                diff = np.clip(diff, -0.1, 0.1)
                control_data.wheels_position = (
                    read_data.joints[0] + diff,
                    read_data.joints[1] + diff,
                    read_data.joints[2] - diff,
                    read_data.joints[3] - diff,
                )
            else:
                _, pc_hat, _ = fk(read_data.joints[4:])
                theta_x = np.pi - R.from_quat(pc_hat[3:]).as_euler("xyz")[0]
                dist = pc_hat[2] * np.tan(theta_x)
                if dist > 0.16:
                    dist = np.clip(dist, -0.1, 0.1)
                    control_data.wheels_position = (
                        read_data.joints[0] - dist,
                        read_data.joints[1] - dist,
                        read_data.joints[2] - dist,
                        read_data.joints[3] - dist,
                    )
                else:
                    return True

    #
    # target 상단에 로봇팔이 위치하도록 ik 계산
    #
    def _calc_ik_ee_target(self, context: Context, read_data: ReadData, control_data: ControlData):
        bbox = self._detect_red_box(read_data.img)
        if not bbox:
            return
        _, pc_hat, _ = fk(read_data.joints[4:])
        theta_x = np.pi - R.from_quat(pc_hat[3:]).as_euler("xyz")[0]
        dist = pc_hat[2] * np.tan(theta_x)
        pt_hat = np.array([pc_hat[0] + dist - 0.025, pc_hat[1], 0.02])
        result = solve_ee(read_data.joints[4:], pt_hat)

        context.manipulator_control_target = (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
        )
        return True

    #
    # 로봇팔을 이용하여 Target을 Pick 하여 적재함에 런칭
    #
    def pick_target(self, context: Context, read_data: ReadData, control_data: ControlData):
        if context.state_count == 1:
            read_data.scan_flg = False
            read_data.img_flag = True

            context.mainpulator_state = 1
        elif context.mainpulator_state == 1:  # calc ik for target
            if self._calc_ik_ee_target(context, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 2:  # control joint for target
            if self._control_joint(context.manipulator_control_target, read_data, control_data):
                context.manipulator_control_target = []
                context.mainpulator_state += 1
        elif 2 < context.mainpulator_state < 13:  # grip target
            control_data.gripper = True
            context.mainpulator_state += 1
        elif context.mainpulator_state == 13:  # load target on loading box
            manipulator_control_target = (
                np.deg2rad(0),
                np.deg2rad(45),
                np.deg2rad(45),
                np.deg2rad(55),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                return True

    def place_target(self, context: Context, read_data: ReadData, control_data: ControlData):
        if context.state_count == 1:
            read_data.scan_flg = False
            read_data.img_flag = True

            context.mainpulator_state = 1
        elif context.mainpulator_state == 1:
            manipulator_control_target = (
                np.deg2rad(0),
                np.deg2rad(-55),
                np.deg2rad(-90),
                np.deg2rad(-25),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif 1 < context.mainpulator_state < 12:
            control_data.gripper = False
            context.mainpulator_state += 1
        elif context.mainpulator_state == 12:
            manipulator_control_target = (
                np.deg2rad(0),
                np.deg2rad(45),
                np.deg2rad(-120),
                np.deg2rad(-25),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 13:
            manipulator_control_target = (
                np.deg2rad(0),
                np.deg2rad(45),
                np.deg2rad(-120),
                np.deg2rad(-60),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 14:
            return True


# forward kinematics
def fk(thetas):
    j0, j1, j2, j3, j4 = thetas[:5]

    # 자동차 -> joint-0
    TC0 = np.array(
        [  # 좌표이동
            [1, 0, 0, 0.166],
            [0, 1, 0, 0],
            [0, 0, 1, 0.099],
            [0, 0, 0, 1],
        ]
    ) @ np.array(
        [  # z 축을 기준으로 j0 회전
            [np.cos(j0), -np.sin(j0), 0, 0],
            [np.sin(j0), np.cos(j0), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # joint-0 -> joint-1
    T01 = np.array(
        [  # 좌표이동 및 x축을 기준으로 90도 회전
            [1, 0, 0, 0.033],
            [0, np.cos(PI_HALF), -np.sin(PI_HALF), 0],
            [0, np.sin(PI_HALF), np.cos(PI_HALF), 0.147],
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
            [1, 0, 0, 0],
            [0, 1, 0, 0.155],
            [0, 0, 1, 0],
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
            [1, 0, 0, 0],
            [0, 1, 0, 0.135],
            [0, 0, 1, 0],
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
    T34 = np.array(
        [  # 좌표이동 및 x축을 기준으로 -90도 회전
            [1, 0, 0, 0.0],
            [0, np.cos(-PI_HALF), -np.sin(-PI_HALF), 0.081],
            [0, np.sin(-PI_HALF), np.cos(-PI_HALF), 0.0],
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
    oe_hat = R.from_matrix(TC4[:-1, :-1]).as_quat()

    pc_hat = TC4 @ np.array([0.0, 0.0, 0.075, 1])
    TCC = TC4 @ np.array(
        [  # z축을 기준으로 90도 회전
            [np.cos(PI_HALF), -np.sin(PI_HALF), 0, 0],
            [np.sin(PI_HALF), np.cos(PI_HALF), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    oc_hat = R.from_matrix(TCC[:-1, :-1]).as_quat()

    p3_hat = TC3 @ np.array([0.0, 0.0, 0.0, 1])
    o3_hat = R.from_matrix(TC3[:-1, :-1]).as_quat()

    return (
        np.concatenate((pe_hat[:3], oe_hat)),
        np.concatenate((pc_hat[:3], oc_hat)),
        np.concatenate((p3_hat[:3], o3_hat)),
    )


#
# ik
#
def ik_ee(thetas, pt):
    pt_hat, _, _ = fk(thetas)
    error = np.linalg.norm(pt[:3] - pt_hat[:3])
    return error


#
# solve
#
def solve_ee(thetas, pt):
    initial_thetas = np.array([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4]])
    theta_bounds = [
        (np.deg2rad(-180), np.deg2rad(180)),
        (np.deg2rad(-75), np.deg2rad(75)),
        (np.deg2rad(-131), np.deg2rad(131)),
        (np.deg2rad(-102), np.deg2rad(-15)),
        (np.deg2rad(-90), np.deg2rad(90)),
    ]

    result = minimize(
        ik_ee,  # 목적 함수
        initial_thetas,  # 초기값
        args=(pt,),  # 추가 매개변수
        bounds=theta_bounds,  # 범위 제한
        method="L-BFGS-B",  # 제약 조건을 지원하는 최적화 알고리즘
        options={"ftol": 1e-9},  # 수렴 기준
    )
    return result.x
