# Copyright 2024 @With-Robot 3rd
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import cv2

from util import State, Config, Context, ReadData, ControlData


# pi / 2
PI_HALF = np.pi / 2

config: Config = Config()
visual_objects = [None] * 128
TARGET = np.array([[49.0, 137.0], [110.0, 139.0], [109.0, 193.0], [52.0, 191.0], [159.0, 87.0], [223.0, 90.0], [214.0, 150.0], [154.0, 148.0]])

# find target
def find_target(context: Context, youbot_data: ReadData):
    result: bool = False
    control_data: ControlData = ControlData()

    # move joint to init position
    if context.state_counter == 1:
        context.mainpulator_state = 0
        context.target_location = None
        control_data.joints_position = config.find_target_init.copy()
        control_data.control_cb = find_target_cb
        control_data.read_camera = True
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
        if context.state == State.PickTarget and state == 2:
            pass
        else:
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
                    _, pc_hat, rpy = fk(youbot_data.joints[1:], [youbot_data.joints[0]])
                    xyz = R.from_quat(pc_hat[3:]).as_euler("xyz")
                    theta_z = np.pi + xyz[0]
                    theta_x = xyz[2]
                    dist = pc_hat[2] * np.tan(theta_z)
                    pt_hat = pc_hat[:3] + dist * np.array(
                        [-np.sin(theta_x), np.cos(theta_x), 0]
                    )
                    pt_hat[-1] = config.target_height
                    context.target_location = pt_hat
                    return True

# move car near to target location
def approach_to_target(context: Context, youbot_data: ReadData):
    result: bool = False
    control_data: ControlData = ControlData()

    if context.state_counter == 1:
        control_data.control_cb = approach_to_target_cb
        control_data.delta = 0.05
        control_data.read_camera = True
    else:
        result = True

    return result, control_data


def approach_to_target_cb(
    context: Context, youbot_data: ReadData, control_data: ControlData
):
    visualize(youbot_data.img)

    # control arm joints
    bboxs = detect_red_box(youbot_data.img)
    if bboxs:
        near_index = find_nearest_bbox(youbot_data.img, bboxs)
        target_location = bboxs[near_index]

        base_x = youbot_data.img.shape[1] // 2
        base_y = youbot_data.img.shape[0] // 2
        x, y, w, h = target_location
        center_x = x + w // 2
        center_y = y + h // 2
        diff_x = center_x - base_x
        diff_y = center_y - base_y

        control_data.joints_position = youbot_data.joints[:5]
        control_data.joints_position[0] -= diff_x / 1000
        control_data.joints_position[1] -= diff_y / 1000

        _, pc_hat, _ = fk(youbot_data.joints[1:], [youbot_data.joints[0]])
        xyz = R.from_quat(pc_hat[3:]).as_euler("xyz")
        theta_z = np.pi + xyz[0]
        theta_x = xyz[2]
        dist = pc_hat[2] * np.tan(theta_z)
        pt_hat = pc_hat[:3] + dist * np.array([-np.sin(theta_x), np.cos(theta_x), 0])
        pt_hat[1] -= 0.01
        pt_hat[-1] = config.target_height
        context.target_location = pt_hat

    # control wheel joints
    angle = np.arctan2(context.target_location[0], context.target_location[1])
    if abs(angle) > 0.003:  # rotate
        control_data.wheels_position = [
            youbot_data.wheels[0] - angle,
            youbot_data.wheels[1] - angle,
            youbot_data.wheels[2] + angle,
            youbot_data.wheels[3] + angle,
        ]
    elif context.target_location[1] > 0.52:  # move forward
        delta = np.clip(abs(context.target_location[1] - 0.52), 0.05, 0.1)
        control_data.wheels_position = [
            youbot_data.wheels[0] - delta,
            youbot_data.wheels[1] - delta,
            youbot_data.wheels[2] - delta,
            youbot_data.wheels[3] - delta,
        ]
    else:
        return True


# pick target
def pick_target(context: Context, youbot_data: ReadData, f_len : float):
    result: bool = False
    control_data: ControlData = ControlData()

    if context.state_counter == 1:
        context.mainpulator_state = 0
    if context.mainpulator_state == 2:
        # pass
        if len(TARGET) == len(context.pixelPositions):
            pixel_error = np.linalg.norm(TARGET - np.array(context.pixelPositions))
            print("pixel error",pixel_error)
            if pixel_error < config.ibvs_threshold:
                context.mainpulator_state += 1
    else:
        context.mainpulator_state += 1

    visualize(youbot_data.img)
    #### this section will be changed by path planning ####
    # move joint to above of target position
    if context.mainpulator_state == 1:
        pt_hat = context.target_location.copy()
        pt_hat[-1] = config.target_height
        target_thetas = solve(youbot_data.joints, pt_hat)
        control_data.joints_position = target_thetas
        context.tmp_target_theta = target_thetas
        print("target theta",np.round(np.degrees(target_thetas),2))
    #### this section will be changed by visual servoing ####
    # move joint to target position
    elif context.mainpulator_state == 2:
        control_data.delta = 0.005
        tmp_target_theta = context.tmp_target_theta
        # read youbot data
        bgr_img = youbot_data.img[:, :, ::-1]
        youbotJoints = youbot_data.joints
        # detect feature
        pixelPositions = detect_features(bgr_img)
        context.pixelPositions = pixelPositions
        if len(pixelPositions) < 8:
            pt_hat = context.target_location.copy()
            pt_hat[-1] = config.target_height
            target_thetas = solve(youbot_data.joints, pt_hat)
            control_data.joints_position = target_thetas
            context.tmp_target_theta = target_thetas
        else:
            pt_hat = calculateTarget(pixelPositions, youbotJoints, f_len)
            target_thetas = solve_close(youbot_data.joints, pt_hat, tmp_target_theta)
            control_data.joints_position = target_thetas
            context.tmp_target_theta = target_thetas
            print("ibvs target theta",np.round(np.degrees(target_thetas),2))
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
    #### this section will be changed by path planning ####
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
    ########################################################
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
    oc_hat = R.from_matrix(TC4[:-1, :-1]).as_quat() # camera quaternion
    rpy = R.from_matrix(TC4[:-1, :-1]).as_euler('xyz') # camera orientation
    rpy[2] = -rpy[2]
    return pe_hat[:3], np.concatenate((pc_hat[:3], oc_hat)), rpy


def ik(thetas, params):
    pt = params[-1][:3]
    pe_hat, _, _ = fk(thetas, params)
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

def ik_cost(thetas, pt):
    params = [thetas[0], pt]
    t_pose = pt[:3]
    t_quat = pt[3:]
    _, c_hat, cam_orien = fk(thetas[1:], params)
    cam_pose = c_hat[:3]
    cam_quat = R.from_euler('xyz', cam_orien).as_quat()
    # 포즈와 방향의 오차를 비용으로 계산
    position_error = np.linalg.norm((t_pose - cam_pose))
    orientation_error = np.linalg.norm(t_quat - cam_quat)
    return position_error + orientation_error


def solve(js, pt):
    target_thetas = fsolve(ik, [js[1], js[2], js[3], js[4]], [js[0], pt])
    target_thetas[3] = 0
    return np.concatenate((np.array([js[0]]), target_thetas))

def solve_close(js, pt, tmp_target_theta):
    initial_thetas = np.array([js[0], js[1], js[2], js[3], js[4]])
    theta_bounds = [(tmp_target_theta[0],tmp_target_theta[0]),
                    (np.deg2rad(-90), np.deg2rad(-30)),
                    (np.deg2rad(-90), np.deg2rad(-10)),
                    (np.deg2rad(-90), np.deg2rad(-30)),
                    (np.deg2rad(-45),np.deg2rad(45))]
    
    result = minimize(
        ik_cost,                     # 목적 함수
        initial_thetas,              # 초기값
        args=(pt),              # 추가 매개변수
        bounds=theta_bounds,         # 범위 제한
        method='L-BFGS-B',           # 제약 조건을 지원하는 최적화 알고리즘
        options={'ftol': 1e-9}       # 수렴 기준
    )
    return result.x

def detect_features(image):
    # BGR에서 HSV로 변환
    image = image.copy()
    image = cv2.flip(image,0)
    image = cv2.flip(image,1)
    pixelPositions = []
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
        if len(ids) >= 2:
            ids_array = ids.flatten()
            # argsort()를 사용해 인덱스 얻기
            sorted_indices = np.argsort(ids_array)
            ids = ids[sorted_indices]
            corners = [corners[i] for i in sorted_indices]
            for corner in corners:
                corner = np.squeeze(corner)
                for feat in corner:
                    x,y = feat
                    pixelPositions.append([x, y])
    return pixelPositions
        
def getImageJacobian(z, u, v, f_len):
    img_jacobian = np.array([[-f_len/z, 0, u/z, (u*v)/f_len, -f_len - (u**2)/f_len, v],
        [0, -f_len/z, v/z, f_len + (v**2)/f_len, -(u*v)/f_len, -u]])
    return img_jacobian

def CalculateJacobian(PixelPositions, cam_position, f_len):
    x,y,z = cam_position
    img_jacobians = []
    for position,t_position in zip(PixelPositions, TARGET):
        u,v = position
        tu,tv = t_position
        img_jacobian = getImageJacobian(z, u, v, f_len)
        img_jacobians.append(img_jacobian)
    if img_jacobians:
        img_jacobians = np.concatenate(img_jacobians, axis=0)
    return img_jacobians

def calculateTarget(pixelPositions, joints, f_len):
    # get camera localization
    _, c_hat, cam_orien = fk(joints[1:5], [joints[0]])
    targetPose = np.concatenate((c_hat[:3], cam_orien))
    # update target camera localization
    cam_position = targetPose[:3]
    J = CalculateJacobian(pixelPositions, cam_position, f_len)
    if len(J) > 0:
        velPixel = (TARGET - np.array(pixelPositions)).flatten()
        Jpinv = np.linalg.pinv(J)
        velCam = Jpinv@velPixel
        # print("output",velCam[:3],np.degrees(velCam[3:]))
        x, y, z = velCam[:3]/30
        a, b, c = velCam[3:]*1.2
        targetPose[0] += x
        targetPose[1] += y
        targetPose[2] -= z
        targetPose[3] += np.clip(a,-0.01, 0.01)
        # targetPose[4] += np.clip(b,-0.1, 0.1)
        targetPose[5] += np.clip(c,-0.1, 0.1)
        for i in range(3,6):
            if targetPose[i] >= np.pi:
                targetPose[i] -= 2*np.pi
            elif targetPose[i] < -np.pi:
                targetPose[i] += 2*np.pi
        pt_quat = R.from_euler('xyz', targetPose[3:]).as_quat()
        pt_hat = np.concatenate((targetPose[:3], pt_quat))
        return pt_hat