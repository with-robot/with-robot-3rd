# Copyright 2024 @With-Robot 3.5
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

import time
import numpy as np
import matplotlib.pyplot as plt

from util import Config, Context

from pynput import keyboard
from pynput.keyboard import Listener


class TestPlanning:
    def __init__(self):
        self.config = Config()
        self.context = Context()
        self.context.map = self.read_map()
        self.start = (5, 5)
        self.end = (95, 95)

        self.MAP_R, self.MAP_P = np.meshgrid(
            np.linspace(-5, 5, self.context.map.shape[0] + 1),
            np.linspace(-5, 5, self.context.map.shape[1] + 1),
        )
        self.plt_objs = [None] * 4096

        self.run_flag = True

    #
    # q 키를 입력하면 실행 종료
    #
    def on_press(self, key):
        if key == keyboard.KeyCode.from_char("q"):
            self.run_flag = False

    #
    # mapping에 생성한 맵을 읽어오는 기능
    # 현재는 임의 생성
    #
    def read_map(self):
        with open("mapping.npy", "rb") as f:
            map = np.load(f)
        return map >= 0

    #
    # wall 기준 0.3미터 거리를 masking
    #
    def make_map_mask(self, map):
        # make mask
        n_row, n_col = map.shape
        walls = np.argwhere(map == 1)
        masks = np.zeros_like(map)
        masks[:3, :] = 1
        masks[-3:, :] = 1
        masks[:, :3] = 1
        masks[:, -3:] = 1
        for x, y in walls:
            masks[
                max(x - 3, 0) : min(x + 4, n_row + 1),
                max(y - 3, 0) : min(y + 4, n_col + 1),
            ] = 1
        masks *= map != 1
        map[masks > 0] = 1.0
        return map

    #
    # 벨만최적방정식을 이용한 map value 계산
    #
    def calc_map_value(self, map_mask, end):
        n_row, n_col = map_mask.shape
        v_prev = np.zeros(map_mask.shape)
        v_next = np.zeros(map_mask.shape)

        def cal_value(row, col):
            if map_mask[row, col] == 1.0:
                return -n_row * n_col
            prev_row = max(0, row - 1)
            next_row = min(n_row - 1, row + 1)
            prev_col = max(0, col - 1)
            next_col = min(n_col - 1, col + 1)

            loc_list = [
                (prev_row, col),  # up
                (next_row, col),  # down
                (row, prev_col),  # left
                (row, next_col),  # right
                (prev_row, prev_col),  # up-left
                (prev_row, next_col),  # up-right
                (next_row, prev_col),  # down-left
                (next_row, next_col),  # down-right
            ]

            loc_value = []
            for loc in loc_list:
                value = -1 + (v_prev[row, col] if map_mask[loc] == 1.0 else v_prev[loc])
                loc_value.append(value)
            return max(loc_value)

        for _ in range(1000):
            v_next.fill(0.0)
            for row in range(n_row):
                for col in range(n_col):
                    if (row, col) == end:
                        pass
                    else:
                        v_next[row, col] = cal_value(row, col)
            if np.sum(np.abs(v_prev - v_next)) < 0.1:
                print("planning ok ...")
                break
            v_prev, v_next = v_next, v_prev

        return v_next

    #
    # map_value 이용한 path 계산
    #
    def calc_map_path(self, map_value, start, end):
        checked = set()
        n_row, n_col = map_value.shape
        position = start
        map_path = []
        for i in range(1000):
            map_path.append(position)
            if position == end:
                break
            row, col = position
            prev_row = max(0, row - 1)
            next_row = min(n_row - 1, row + 1)
            prev_col = max(0, col - 1)
            next_col = min(n_col - 1, col + 1)

            items = [
                (prev_row, col),  # up
                (next_row, col),  # down
                (row, prev_col),  # left
                (row, next_col),  # right
                (prev_row, prev_col),  # up-left
                (prev_row, next_col),  # up-right
                (next_row, prev_col),  # down-left
                (next_row, next_col),  # down-right
            ]
            loc_list = []
            for loc in items:
                if loc not in checked:
                    loc_list.append(loc)
                    checked.add(loc)

            loc_value = np.zeros(len(loc_list))
            for i, loc in enumerate(loc_list):
                loc_value[i] = map_value[loc]

            index = np.argmax(loc_value)
            position = loc_list[index]
        return np.array(map_path)

    #
    # 시각화
    #
    def visualize(self, map, map_path):
        map[map_path[:, 0], map_path[:, 1]] = 0.5

        for i in range(len(self.plt_objs)):
            if self.plt_objs[i] is None:
                break
            self.plt_objs[i].remove()
            self.plt_objs[i] = None

        self.plt_objs[0] = plt.pcolor(self.MAP_R, self.MAP_P, map * -1, cmap="gray")
        plt.axis("off")
        plt.axis("equal")
        plt.pause(0.01)

    def run(self):
        # key input
        Listener(on_press=self.on_press).start()

        # initial data
        map_mask = self.make_map_mask(self.context.map.copy())
        # start = (60, 50)
        start = self.config.place["/belcony_end"]
        end = self.config.place["/bedroom1"]
        # planning
        map_value = self.calc_map_value(map_mask, end)
        map_path = self.calc_map_path(map_value, start, end)

        # visualize
        while self.run_flag:
            self.visualize(self.context.map.copy(), map_path)
            time.sleep(0.1)


if __name__ == "__main__":
    main = TestPlanning()
    # run
    main.run()
