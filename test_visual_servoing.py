# Copyright 2024 @With-Robot 3.5
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

import numpy as np
from itertools import permutations
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import cv2
import matplotlib.pyplot as plt

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from util import Context, ReadData, ControlData

from pynput import keyboard
from pynput.keyboard import Listener


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


class TestVisualServoing:
    def __init__(self):
        # coppeliasim simulation instance
        self.sim = RemoteAPIClient().require("sim")
        # Simulation run flag
        self.run_flag = True

        # Wheel jointsq
        self.joints = []
        self.joints.append(self.sim.getObject("/rollingJoint_fl"))
        self.joints.append(self.sim.getObject("/rollingJoint_rl"))
        self.joints.append(self.sim.getObject("/rollingJoint_fr"))
        self.joints.append(self.sim.getObject("/rollingJoint_rr"))
        # manipulator 5 joints
        for i in range(5):
            self.joints.append(self.sim.getObject(f"/youBotArmJoint{i}"))

        self.camera_1 = self.sim.getObject(f"/camera_1")

        self.context = Context()
        self.read_data = ReadData()
        self.control_data = ControlData()

        self.plt_objs = [None] * 4096

    def on_press(self, key):
        if key == keyboard.KeyCode.from_char("q"):
            self.run_flag = False

    def read_youbot(self):
        # camera location
        p = self.sim.getObjectPosition(self.camera_1)
        o = self.sim.getObjectOrientation(self.camera_1)
        self.read_data.cam_localization = np.array(p + o)

        # read manipulator joints
        joints = []
        for joint in self.joints:
            theta = self.sim.getJointPosition(joint)
            joints.append(theta)
        self.read_data.joints = np.array(joints)

        # camera image
        result = self.sim.getVisionSensorImg(self.camera_1)
        img = np.frombuffer(result[0], dtype=np.uint8)
        img = img.reshape((result[1][1], result[1][0], 3))
        img = cv2.flip(img, 0)
        self.read_data.img = img

    def control_youbot(self):
        if self.control_data.manipulator_position is not None:
            for i, joint in enumerate(self.control_data.manipulator_position):
                index = 4 + i
                diff = abs(joint - self.read_data.joints[index])
                diff = min(diff, np.pi / 2)
                if self.read_data.joints[index] < joint:
                    target = self.read_data.joints[index] + diff
                else:
                    target = self.read_data.joints[index] - diff
                self.sim.setJointTargetPosition(self.joints[index], target)

    def visualize(self):
        for i in range(len(self.plt_objs)):
            if self.plt_objs[i] is None:
                break
            self.plt_objs[i].remove()
            self.plt_objs[i] = None

        self.plt_objs[0] = plt.imshow(self.read_data.img)

        plt.axis("off")
        plt.axis("equal")
        plt.pause(0.01)

    def callback(self):
        visual_servoing(self.context, self.read_data)
        self.control_data.manipulator_position = self.context.manipulator_control_target
        self.visualize()

    def run(self):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()

        # execution of the simulation
        while self.run_flag:
            # read youbot data
            self.read_youbot()
            # callback
            self.callback()
            # control youbot
            self.control_youbot()
            # Run Simulation Step
            self.sim.step()

        # Stop Simulation
        self.sim.stopSimulation()


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
# ik cam position
#
def ik_cam_position(thetas, cl):
    _, cl_hat, _ = fk(thetas)
    error = np.linalg.norm(cl[:3] - cl_hat[:3])
    return error


#
# solve cam position
#
def solve_cam_position(thetas, cl):
    initial_thetas = np.array([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4]])
    theta_bounds = [
        (np.deg2rad(-180), np.deg2rad(180)),
        (np.deg2rad(-75), np.deg2rad(75)),
        (np.deg2rad(-131), np.deg2rad(131)),
        (np.deg2rad(-102), np.deg2rad(-15)),
        (np.deg2rad(-90), np.deg2rad(90)),
    ]

    result = minimize(
        ik_cam_position,  # 목적 함수
        initial_thetas,  # 초기값
        args=(cl,),  # 추가 매개변수
        bounds=theta_bounds,  # 범위 제한
        method="L-BFGS-B",  # 제약 조건을 지원하는 최적화 알고리즘
        options={"ftol": 1e-9},  # 수렴 기준
    )
    return (
        result.x[0],
        result.x[1],
        result.x[2],
        result.x[3],
    )


#
# ik cam orientation
#
def ik_cam_orientation(thetas, cl):
    _, cl_hat, _ = fk(thetas)
    error = np.linalg.norm(cl[5:] - cl_hat[5:])
    return error


#
# solve cam orientation
#
def solve_cam_orientation(thetas, cl):
    initial_thetas = np.array([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4]])
    theta_bounds = [
        (thetas[0], thetas[0]),
        (thetas[1], thetas[1]),
        (thetas[2], thetas[2]),
        (thetas[3], thetas[3]),
        (np.deg2rad(-90), np.deg2rad(90)),
    ]

    result = minimize(
        ik_cam_orientation,  # 목적 함수
        initial_thetas,  # 초기값
        args=(cl,),  # 추가 매개변수
        bounds=theta_bounds,  # 범위 제한
        method="L-BFGS-B",  # 제약 조건을 지원하는 최적화 알고리즘
        options={"ftol": 1e-9},  # 수렴 기준
    )
    return (result.x[4],)


#
# visual servoing을 위한 feature detection
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


def ibvs_jacobian(pixel, Z, focal_alpha):
    x, y = pixel / focal_alpha
    return np.array(
        [
            [-1 / Z, 0, x / Z, x * y, -(1 + x**2), y],
            [0, -1 / Z, y / Z, 1 + y**2, -x * y, -x],
        ]
    )


def match_ibvs_pixels(pixel_positions, refer_positions):
    min_norm, min_pixels = 1e9, None
    for pixels in permutations(pixel_positions):
        pixels = np.array(pixels)
        norm = np.linalg.norm(pixels - refer_positions)
        if norm < min_norm:
            min_norm = norm
            min_pixels = pixels
    return min_pixels


def visual_servoing(context: Context, read_data: ReadData):
    pixel_positions = detect_ibvs_features(read_data.img[:, :, ::-1])
    if len(pixel_positions) == len(TARGET):
        pixel_positions = match_ibvs_pixels(pixel_positions, TARGET)

        if True:
            _, height, _ = read_data.img.shape
            for i, p in enumerate(TARGET):
                cv2.circle(read_data.img, (int(p[0]), height - int(p[1])), 2, (0, 0, 255), -1)
            for i, p in enumerate(pixel_positions):
                cv2.circle(read_data.img, (int(p[0]), height - int(p[1])), 2, (0, 255, 0), -1)

        pixel_positions = pixel_positions - CENTER
        refer_positions = TARGET - CENTER

        _, cl_hat, _ = fk(read_data.joints[4:])
        # cl = read_data.cam_localization
        Z = cl_hat[2] - 0.04
        Z_ref = TARGET_Z - 0.04

        lamda = 0.001
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
        # control_sum = np.clip(-0.05, 0.05, control_sum)

        cl_position = cl_hat.copy()
        cl_position[0] += control_sum[1]
        cl_position[1] -= control_sum[0]
        cl_position[2] -= control_sum[2]
        cl_position[3] = -np.pi
        cl_position[4] = 0.0
        position = solve_cam_position(read_data.joints[4:], cl_position)

        cl_orientation = cl_hat.copy()
        cl_orientation[3] = -np.pi
        cl_orientation[4] = 0.0
        cl_orientation[5] += control_sum[5] * 100
        orientation = solve_cam_orientation(read_data.joints[4:], cl_orientation)

        context.manipulator_control_target = (
            position[0],
            position[1],
            position[2],
            position[3],
            orientation[0],
        )


if __name__ == "__main__":
    main = TestVisualServoing()
    # run
    main.run()
