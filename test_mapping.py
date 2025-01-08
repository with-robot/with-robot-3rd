# Copyright 2024 @With-Robot 3.5
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

import numpy as np
import matplotlib.pyplot as plt

from coppeliasim import Coppeliasim

from util import Config, Context, ReadData, ControlData

from pynput import keyboard
from pynput.keyboard import Key, Listener


class TestMapping:
    def __init__(self):
        self.sim = Coppeliasim()

        self.config = Config()
        self.context = Context()
        # map
        self.context.map = np.zeros(self.config.map_size)
        # location of map grid
        self.grid_loc = np.zeros((self.config.map_size[0], self.config.map_size[1], 2))
        full = self.config.map_size[0] * 0.1
        self.grid_loc[:, :, 0] = np.linspace(-full / 2 + 0.05, full / 2 - 0.05, self.config.map_size[0]).reshape(1, -1)
        full = self.config.map_size[1] * 0.1
        self.grid_loc[:, :, 1] = np.linspace(-full / 2 + 0.05, full / 2 - 0.05, self.config.map_size[1]).reshape(-1, 1)
        # control signal
        self.vel = 0.0
        self.rot = 0.0
        # visualize
        self.MAP_R, self.MAP_P = np.meshgrid(np.linspace(-5, 5, 101), np.linspace(-5, 5, 101))
        self.plt_objs = [None] * 4096

        self.sim.read_data.scan_flg = True

    def on_press(self, key):
        if key == Key.up:
            self.vel += 1
            self.rot += 1 if self.rot < 0 else -1
        if key == Key.down:
            self.vel -= 1
            self.rot += 1 if self.rot < 0 else -1
        if key == Key.left:
            self.rot += 1
            self.vel += 1 if self.vel < 0 else -1
        if key == Key.right:
            self.rot -= 1
            self.vel += 1 if self.vel < 0 else -1
        self.vel = np.clip(self.vel, -50, 50)
        self.rot = np.clip(self.rot, -10, 10)

        if key == keyboard.KeyCode.from_char("q"):
            with open("mapping.npy", "wb") as f:
                np.save(f, self.context.map)
            self.sim.run_flag = False

    def mapping(self, read_data: ReadData, visualize=True):
        n_row, n_col = self.config.map_size
        # car position
        c_x, c_y, _ = read_data.localization[:3]
        _, _, c_z = read_data.localization[3:]
        # lidar positionq
        l_x = c_x + self.config.lidar_offset * np.cos(c_z)
        l_y = c_y + self.config.lidar_offset * np.sin(c_z)
        l_point = np.array([l_x, l_y])
        # lidar sensing points (일반 lidar와 동작 방식이 다름)
        r_points = read_data.scan.reshape(-1, 2)
        s_points = r_points - l_point
        s_angles = np.arctan2(s_points[:, 1], s_points[:, 0])
        s_dists = np.linalg.norm(s_points, axis=-1)
        # distance and angle
        g_points = self.grid_loc - l_point
        g_dists = np.linalg.norm(g_points, axis=-1)
        g_angles = np.stack(
            [
                np.arctan2(g_points[:, :, 1] + 0.05, g_points[:, :, 0] + 0.05),  # up-right
                np.arctan2(g_points[:, :, 1] - 0.05, g_points[:, :, 0] + 0.05),  # down-right
                np.arctan2(g_points[:, :, 1] + 0.05, g_points[:, :, 0] - 0.05),  # up-left
                np.arctan2(g_points[:, :, 1] - 0.05, g_points[:, :, 0] - 0.05),  # down-left
            ],
            axis=-1,
        )
        g_angles_max = np.max(g_angles, axis=-1)
        g_angles_min = np.min(g_angles, axis=-1)
        g_angles_mask = (g_angles_max - g_angles_min) > np.pi

        # diff angles
        occupied = set()
        free = set()
        for i in range(len(s_angles)):
            point = r_points[i]
            norm = np.linalg.norm(self.grid_loc - point, axis=-1)
            index = np.argmin(norm.reshape(-1))
            x, y = index // n_row, index % n_col

            dist = s_dists[i]
            if dist < 4.99:
                occupied.add((x, y))

            angle = s_angles[i]
            dist = s_dists[i]
            angle_valid = ((g_angles_min < angle) * (angle < g_angles_max)) == 1
            angle_valid[g_angles_mask] = np.sum(angle_valid) == 0
            dist_valid = g_dists < dist
            dist_valid[x, y] = 0
            index = np.where(angle_valid * dist_valid == 1)
            for x, y in zip(index[0], index[1]):
                free.add((x, y))

        free = np.array(list(free - occupied))
        occupied = np.array(list(occupied))
        self.context.map[free[:, 0], free[:, 1]] -= 0.5
        self.context.map[occupied[:, 0], occupied[:, 1]] += 0.5
        np.clip(self.context.map, -10, 10, self.context.map)

        # visualize
        if visualize:
            for i in range(len(self.plt_objs)):
                if self.plt_objs[i] is None:
                    break
                self.plt_objs[i].remove()
                self.plt_objs[i] = None

            self.plt_objs[0] = plt.pcolor(self.MAP_R, self.MAP_P, -self.context.map, cmap="gray")
            (self.plt_objs[1],) = plt.plot(c_x, c_y, color="green", marker="o", markersize=10)
            (self.plt_objs[2],) = plt.plot([c_x, l_x], [c_y, l_y], "-b")

            draw = np.concatenate(
                (
                    np.array([[l_x, l_y]]),
                    r_points,
                    np.array([[l_x, l_y]]),
                )
            )
            (self.plt_objs[3],) = plt.fill(draw[:, 0], draw[:, 1], "red", alpha=0.75)
            plt.axis("off")
            plt.axis("equal")
            plt.pause(0.01)

    def callback(self, read_data: ReadData, control_data: ControlData):
        if read_data.scan is not None:
            self.mapping(read_data)

        control_data.wheels_position = np.array(
            [
                read_data.joints[0] - (self.vel - self.rot) * 0.1,
                read_data.joints[1] - (self.vel - self.rot) * 0.1,
                read_data.joints[2] - (self.vel + self.rot) * 0.1,
                read_data.joints[3] - (self.vel + self.rot) * 0.1,
            ]
        )
        # decrase speed
        if self.vel != 0:
            self.vel += 1 if self.vel < 0 else -1
        if self.rot != 0:
            self.rot += 1 if self.rot < 0 else -1

    def run(self):
        # key input
        Listener(on_press=self.on_press).start()
        # run sim
        self.sim.run(self.callback)


if __name__ == "__main__":
    main = TestMapping()
    # run
    main.run()
