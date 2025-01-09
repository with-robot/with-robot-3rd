# Copyright 2024 @With-Robot 3.5
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

import numpy as np
from itertools import permutations
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import cv2

from util import Config, Context, ReadData, ControlData


PI_HALF = np.pi / 2
CENTER = np.array([[127.5, 128.5]])
TARGET_Z = 0.08268
TARGET = np.array(
    [
        [178.0, 78.0],
        [78.0, 78.0],
        [78.0, 178.0],
        [178.0, 178.0],
    ]
)
FOCAL_ALPHA = 223.0


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
        control_data.wheels_position = tuple(read_data.joints[:4])

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
                _, cl_hat, _ = fk(read_data.joints[4:])
                co_x = np.pi - cl_hat[3]
                dist = cl_hat[2] * np.tan(co_x)
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
    # target 상단에 joint3이 위치하도록 ik 계산
    #
    def _calc_ik_j3_target(self, context: Context, read_data: ReadData, control_data: ControlData):
        bbox = self._detect_red_box(read_data.img)
        if not bbox:
            return
        _, cl_hat, _ = fk(read_data.joints[4:])
        co_x = np.pi - cl_hat[3]
        dist = cl_hat[2] * np.tan(co_x)
        tl_hat = np.array([cl_hat[0] + dist - 0.025, cl_hat[1], 0.30, np.pi, 0, -np.pi])
        result = solve_j3(read_data.joints[4:], tl_hat)

        context.manipulator_control_target = (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
        )
        return True

    #
    # target ee가 위치하도록 ik 계산
    #
    def _calc_ik_ee_target(self, context: Context, read_data: ReadData, control_data: ControlData):
        bbox = self._detect_red_box(read_data.img)
        if not bbox:
            return
        _, cl_hat, _ = fk(read_data.joints[4:])
        co_x = np.pi - cl_hat[3]
        dist = cl_hat[2] * np.tan(co_x)
        tl_hat = np.array([cl_hat[0] + dist, cl_hat[1], 0.06, np.pi, 0, -np.pi])
        result = solve_ee(read_data.joints[4:], tl_hat)

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
        elif context.mainpulator_state == 1:  # align bbox center
            bbox = self._detect_red_box(read_data.img)
            if not bbox:
                return
            if self._control_bbox_center(bbox, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 2:  # calc ik for above target
            if self._calc_ik_j3_target(context, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 3:  # control joint for above target
            if self._control_joint(context.manipulator_control_target, read_data, control_data):
                context.manipulator_control_target = []
                context.mainpulator_state += 1
        elif context.mainpulator_state == 4:  # align bbox center
            bbox = self._detect_red_box(read_data.img)
            if not bbox:
                return
            if self._control_bbox_center(bbox, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 5:  # visual servoing
            if visual_servoing(context, read_data):
                context.manipulator_control_target = []
                context.mainpulator_state += 1
            else:
                self._control_joint(context.manipulator_control_target, read_data, control_data)
        elif context.mainpulator_state == 6:  # align bbox center
            bbox = self._detect_red_box(read_data.img)
            if not bbox:
                return
            if self._control_bbox_center(bbox, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 7:  # calc ik for grip target
            if self._calc_ik_ee_target(context, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 8:  # control joint for grip target
            if self._control_joint(context.manipulator_control_target, read_data, control_data):
                context.manipulator_control_target = []
                context.mainpulator_state += 1
        elif 8 < context.mainpulator_state < 18:  # grip target
            control_data.gripper = True
            context.mainpulator_state += 1
        elif context.mainpulator_state == 18:  # pick up target
            manipulator_control_target = (
                np.deg2rad(0),
                np.deg2rad(45),
                np.deg2rad(-90),
                np.deg2rad(-60),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 19:  # rotate onto the cargo
            manipulator_control_target = (
                np.deg2rad(-180),
                np.deg2rad(45),
                np.deg2rad(-90),
                np.deg2rad(-60),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 20:  # place target on the cargo
            manipulator_control_target = (
                np.deg2rad(-180),
                np.deg2rad(-25),
                np.deg2rad(-74),
                np.deg2rad(-81),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif 20 < context.mainpulator_state < 30:  # un grip target
            control_data.gripper = False
            context.mainpulator_state += 1
        elif context.mainpulator_state == 30:  # check cargo state
            manipulator_control_target = (
                np.deg2rad(-180),
                np.deg2rad(-15),
                np.deg2rad(-75),
                np.deg2rad(-82),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 31:  # check cargo state
            bbox = self._detect_red_box(read_data.img)
            if bbox:
                return True
            else:
                context.mainpulator_state = 1

    def place_target(self, context: Context, read_data: ReadData, control_data: ControlData):
        if context.state_count == 1:
            read_data.scan_flg = False
            read_data.img_flag = True

            context.mainpulator_state = 1
        elif context.mainpulator_state == 1:  # pick target on the cargo
            manipulator_control_target = (
                np.deg2rad(-180),
                np.deg2rad(-25),
                np.deg2rad(-74),
                np.deg2rad(-81),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif 1 < context.mainpulator_state < 11:  # grip target
            control_data.gripper = True
            context.mainpulator_state += 1
        elif context.mainpulator_state == 11:  # pick up target
            manipulator_control_target = (
                np.deg2rad(-180),
                np.deg2rad(45),
                np.deg2rad(-120),
                np.deg2rad(-60),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 12:  # rotate fo place
            manipulator_control_target = (
                np.deg2rad(0),
                np.deg2rad(45),
                np.deg2rad(-120),
                np.deg2rad(-60),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 13:
            manipulator_control_target = (
                np.deg2rad(0),
                np.deg2rad(-55),
                np.deg2rad(-90),
                np.deg2rad(-25),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif 13 < context.mainpulator_state < 23:
            control_data.gripper = False
            context.mainpulator_state += 1
        elif context.mainpulator_state == 23:
            manipulator_control_target = (
                np.deg2rad(0),
                np.deg2rad(45),
                np.deg2rad(-120),
                np.deg2rad(-25),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 24:
            manipulator_control_target = (
                np.deg2rad(0),
                np.deg2rad(45),
                np.deg2rad(-120),
                np.deg2rad(-60),
                np.deg2rad(0),
            )
            if self._control_joint(manipulator_control_target, read_data, control_data):
                context.mainpulator_state += 1
        elif context.mainpulator_state == 25:
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

    ep_hat = TC4 @ np.array([0.0, 0.0, 0.123, 1])
    eo_hat = R.from_matrix(TC4[:-1, :-1]).as_euler("xyz")

    cp_hat = TC4 @ np.array([0.0, 0.0, 0.075, 1])
    TCC = TC4 @ np.array(
        [  # z축을 기준으로 90도 회전
            [np.cos(PI_HALF), -np.sin(PI_HALF), 0, 0],
            [np.sin(PI_HALF), np.cos(PI_HALF), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    co_hat = R.from_matrix(TCC[:-1, :-1]).as_euler("xyz")

    j3p_hat = TC3 @ np.array([0.0, 0.0, 0.0, 1])
    j3o_hat = R.from_matrix(TC3[:-1, :-1]).as_euler("xyz")

    return (
        np.concatenate((ep_hat[:3], eo_hat)),
        np.concatenate((cp_hat[:3], co_hat)),
        np.concatenate((j3p_hat[:3], j3o_hat)),
    )


#
# ik
#
def ik_j3(thetas, j3l):
    _, _, j3l_hat = fk(thetas)
    p_error = np.linalg.norm(j3l[:3] - j3l_hat[:3])
    return p_error


#
# solve
#
def solve_j3(thetas, j3l):
    initial_thetas = np.array([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4]])
    theta_bounds = [
        (np.deg2rad(-180), np.deg2rad(180)),
        (np.deg2rad(-75), np.deg2rad(75)),
        (np.deg2rad(-131), np.deg2rad(131)),
        (np.deg2rad(-102), np.deg2rad(-15)),
        (np.deg2rad(-90), np.deg2rad(90)),
    ]

    result = minimize(
        ik_j3,  # 목적 함수
        initial_thetas,  # 초기값
        args=(j3l,),  # 추가 매개변수
        bounds=theta_bounds,  # 범위 제한
        method="L-BFGS-B",  # 제약 조건을 지원하는 최적화 알고리즘
        options={"ftol": 1e-9},  # 수렴 기준
    )
    return (
        result.x[0],
        result.x[1],
        result.x[2],
        -np.pi - (result.x[1] + result.x[2]),
        thetas[4],
    )


#
# ik
#
def ik_ee(thetas, el):
    _, cl_hat, _ = fk(thetas)
    p_error = np.linalg.norm(el[:3] - cl_hat[:3])
    return p_error


#
# solve
#
def solve_ee(thetas, el):
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
        args=(el,),  # 추가 매개변수
        bounds=theta_bounds,  # 범위 제한
        method="L-BFGS-B",  # 제약 조건을 지원하는 최적화 알고리즘
        options={"ftol": 1e-9},  # 수렴 기준
    )
    return (
        result.x[0],
        result.x[1],
        result.x[2],
        result.x[3],
        result.x[4],
    )


#
# imabe based visual servoing을 위한 feature detection
#
def detect_ibvs_features(image):
    # BGR에서 HSV로 변환
    image = image.copy()
    # image = cv2.flip(image, 0)ㅂ
    # image = cv2.flip(image, 1)
    pixel_positions = []
    # Aruco 사전 및 파라미터 설정
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    # 이미지 읽기 (회전된 마커 포함)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aruco 마커 감지
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    # 4. 탐지된 마커 처리
    if ids is not None:
        ids_array = ids.flatten()
        # argsort()를 사용해 인덱스 얻기
        sorted_indices = np.argsort(ids_array)
        ids = ids[sorted_indices]
        corners = [corners[i] for i in sorted_indices]
        for corner in corners:
            corner = np.squeeze(corner)
            for feat in corner:
                x, y = feat
                pixel_positions.append([x, image.shape[1] - y])
    return pixel_positions


#
# imabe based visual servoing을 위한 feature matching
#
def match_ibvs_pixels(pixel_positions, refer_positions):
    min_norm, min_pixels = 1e9, None
    for pixels in permutations(pixel_positions):
        pixels = np.array(pixels)
        norm = np.linalg.norm(pixels - refer_positions)
        if norm < min_norm:
            min_norm = norm
            min_pixels = pixels
    return min_pixels


#
# imabe based visual servoing을 위한 jacobian
#
def ibvs_jacobian(pixel, Z, focal_alpha):
    x, y = pixel / focal_alpha
    return np.array(
        [
            [-1 / Z, 0, x / Z, x * y, -(1 + x**2), y],
            [0, -1 / Z, y / Z, 1 + y**2, -x * y, -x],
        ]
    )


#
# imabe based visual servoing
#
def visual_servoing(context: Context, read_data: ReadData):
    pixel_positions = detect_ibvs_features(read_data.img[:, :, ::-1])
    if len(pixel_positions) == len(TARGET):
        pixel_positions = match_ibvs_pixels(pixel_positions, TARGET)

        pixel_positions = pixel_positions - CENTER
        refer_positions = TARGET - CENTER

        _, cl_hat, _ = fk(read_data.joints[4:])
        Z = cl_hat[2] - 0.04
        Z_ref = TARGET_Z - 0.04

        lamda = 0.1
        controls = []
        for i in range(4):
            s_pixel = pixel_positions[i]
            s_refer = refer_positions[i]
            L_pixel = ibvs_jacobian(pixel_positions[i], Z, FOCAL_ALPHA)
            L_refer = ibvs_jacobian(refer_positions[i], Z_ref, FOCAL_ALPHA)
            L = np.linalg.pinv(0.5 * (L_pixel + L_refer))
            control = -lamda * (L @ (s_pixel - s_refer))
            controls.append(control)
        control_sum = np.sum(controls, axis=0)

        if abs(control_sum[5]) < 5e-4:
            return True
        context.manipulator_control_target = (
            read_data.joints[4],
            read_data.joints[5],
            read_data.joints[6],
            read_data.joints[7],
            read_data.joints[8] - control_sum[5] * 25,  # joint 4 번만 제어
        )
