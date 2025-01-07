# Copyright 2024 @With-Robot 3.5
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

import numpy as np
from itertools import permutations
import cv2
import matplotlib.pyplot as plt

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from pynput import keyboard
from pynput.keyboard import Listener


class TestVisualServoing:
    def __init__(self):
        # coppeliasim simulation instance
        self.sim = RemoteAPIClient().require("sim")
        # Simulation run flag
        self.run_flag = True

        self.camera_1 = self.sim.getObject(f"/camera_1")
        self.cam_localization = None
        self.img = None

        self.center_positions = np.array([[127.5, 128.5]])
        self.focal_alpha = 223.0
        self.refer_positions = np.array(
            [
                [10.0, 96.0],
                [95.0, 166.0],
                [165.0, 81.0],
                [80.0, 11.0],
            ]
        )
        self.control = None

        self.plt_objs = [None] * 4096

    def on_press(self, key):
        if key == keyboard.KeyCode.from_char("q"):
            self.run_flag = False

    def read_youbot(self):
        p = self.sim.getObjectPosition(self.camera_1)
        o = self.sim.getObjectOrientation(self.camera_1)
        self.cam_localization = np.array(p + o)

        result = self.sim.getVisionSensorImg(self.camera_1)
        img = np.frombuffer(result[0], dtype=np.uint8)
        img = img.reshape((result[1][1], result[1][0], 3))
        img = cv2.flip(img, 0)
        self.img = img

    def control_youbot(self):
        if self.control is not None:
            position = list(self.cam_localization[:3].copy())

            self.sim.setObjectPosition(
                self.camera_1,
                (
                    position[0] + self.control[0],
                    position[1] + self.control[1],
                    position[2] - self.control[2],
                ),
            )
            orientation = self.cam_localization[3:]
            self.sim.setObjectOrientation(
                self.camera_1,
                (
                    orientation[0],
                    orientation[1],
                    orientation[2] - self.control[5],
                ),
            )
        self.control = None

    def visualize(self, pixel_positions):
        for i in range(len(self.plt_objs)):
            if self.plt_objs[i] is None:
                break
            self.plt_objs[i].remove()
            self.plt_objs[i] = None

        _, height, _ = self.img.shape
        for i, p in enumerate(self.refer_positions):
            cv2.circle(self.img, (int(p[0]), height - int(p[1])), 2, (0, 0, 255), -1)
        for i, p in enumerate(pixel_positions):
            cv2.circle(self.img, (int(p[0]), height - int(p[1])), 2, (0, 255, 0), -1)

        self.plt_objs[0] = plt.imshow(self.img)

        plt.axis("off")
        plt.axis("equal")
        plt.pause(0.01)

    def callback(self):
        pixel_positions = detect_features(self.img[:, :, ::-1])
        if len(pixel_positions) == len(self.refer_positions):
            pixel_positions = match_pixels(pixel_positions, self.refer_positions)
            self.control = visual_servoing(
                pixel_positions,
                self.refer_positions,
                self.center_positions,
                self.cam_localization[2],
                self.focal_alpha,
            )
        self.visualize(pixel_positions)

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


#
# visual servoing을 위한 feature detection
#
def detect_features(image):
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


def match_pixels(pixel_positions, refer_positions):
    min_norm, min_pixels = 1e9, None
    for pixels in permutations(pixel_positions):
        pixels = np.array(pixels)
        norm = np.linalg.norm(pixels - refer_positions)
        if norm < min_norm:
            min_norm = norm
            min_pixels = pixels
    return min_pixels


def visual_servoing(pixel_positions, refer_positions, center_positions, Z, focal_alpha):
    pixel_positions = pixel_positions - center_positions
    refer_positions = refer_positions - center_positions
    lamda = 0.0025
    controls = []
    for i in range(4):
        s_pixel = pixel_positions[i]
        s_refer = refer_positions[i]
        L_pixel = ibvs_jacobian(pixel_positions[i], Z, focal_alpha)
        L_refer = ibvs_jacobian(refer_positions[i], Z, focal_alpha)
        L = np.linalg.pinv(0.5 * (L_pixel + L_refer))
        control = -lamda * (L @ (s_pixel - s_refer))
        controls.append(control)
    control_sum = np.mean(controls, axis=0)
    control_sum[3:] *= 25  # 회전에 좀 더 많은 가중치 적용
    return control_sum


if __name__ == "__main__":
    main = TestVisualServoing()
    # run
    main.run()
