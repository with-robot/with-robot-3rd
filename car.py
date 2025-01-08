# Copyright 2024 @With-Robot 3.5
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

from threading import Thread
import numpy as np

from util import Config, Context, ReadData, ControlData


#
# A class for car operation
#
class CarClass:
    def __init__(self, shape):
        self.config = Config()
        self.map_loc = self._build_map_loc(shape)

    #
    # grid의 좌표 계산
    #
    def _build_map_loc(self, shape):
        map_loc = np.zeros((shape[0], shape[1], 2))
        full = shape[0] * 0.1
        map_loc[:, :, 0] = np.linspace(-full / 2 + 0.05, full / 2 - 0.05, shape[0]).reshape(1, -1)
        full = shape[1] * 0.1
        map_loc[:, :, 1] = np.linspace(-full / 2 + 0.05, full / 2 - 0.05, shape[1]).reshape(-1, 1)
        return map_loc

    #
    # wall 기준 0.3미터 거리를 masking
    #
    def _calc_map_mask(self, map):
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
    def _calc_map_value(self, map_mask, end):
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
    def _calc_map_path(self, context, start, end):
        map_mask = self._calc_map_mask(context.map.copy())
        map_value = self._calc_map_value(map_mask, end)

        checked = set()
        n_row, n_col = map_value.shape
        position = start
        map_path = []
        for i in range(1000):
            map_path.append(tuple(position))
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
        return map_path

    def _calc_map_path_cb(self, context, start, end):
        context.path = self._calc_map_path(context, start, end)
        context.path_idx = 0

    def _calc_angle_diff(self, target, curr):
        angle = target - curr
        if angle < -np.pi:
            angle += 2 * np.pi
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def _follow_path(self, context: Context, read_data: ReadData, control_data: ControlData):
        if len(context.path) <= context.path_idx:
            context.path = None
            context.path_idx = None
            control_data.wheels_position = (
                read_data.joints[0],
                read_data.joints[1],
                read_data.joints[2],
                read_data.joints[3],
            )
            return True

        target = context.path[context.path_idx]
        curr = self.point_to_gird(read_data.localization[:2])
        if target == curr:
            context.path_idx += 1
        else:
            diff = self.map_loc[target] - read_data.localization[:2]

            curr_z = read_data.localization[5]
            target_z = np.arctan2(diff[1], diff[0])
            angle = self._calc_angle_diff(target_z, curr_z)
            distance = np.linalg.norm(diff)
            if abs(angle) > 0.3:
                angle *= 0.5
                control_data.wheels_position = (
                    read_data.joints[0] + angle,
                    read_data.joints[1] + angle,
                    read_data.joints[2] - angle,
                    read_data.joints[3] - angle,
                )
            else:
                distance *= np.pi
                control_data.wheels_position = (
                    read_data.joints[0] - distance,
                    read_data.joints[1] - distance,
                    read_data.joints[2] - distance,
                    read_data.joints[3] - distance,
                )

    #
    # x, y 좌표를 grid 위치로 변환
    #
    def point_to_gird(self, point):
        n_row, n_col = self.map_loc.shape[:2]
        norm = np.linalg.norm(self.map_loc - point, axis=-1)
        index = np.argmin(norm.reshape(-1))
        row, col = index // n_row, index % n_col
        return row, col

    #
    # pick 위치로 이동
    #
    def move_to_pick(self, context: Context, read_data: ReadData, control_data: ControlData):
        if context.state_count == 1:
            read_data.scan_flg = True
            read_data.img_flag = False

            start = self.point_to_gird(read_data.localization[:2])
            end = self.config.place[context.mission.pick_location]
            t = Thread(target=self._calc_map_path_cb, args=(context, start, end))
            t.start()
        elif context.path:
            return self._follow_path(context, read_data, control_data)

    #
    # place 위치로 이동
    #
    def move_to_place(self, context: Context, read_data: ReadData, control_data: ControlData):
        if context.state_count == 1:
            read_data.scan_flg = True
            read_data.img_flag = False

            start = self.point_to_gird(read_data.localization[:2])
            end = self.config.place[context.mission.place_location]
            t = Thread(target=self._calc_map_path_cb, args=(context, start, end))
            t.start()
        elif context.path:
            return self._follow_path(context, read_data, control_data)

    #
    # base 위치로 이동
    #
    def move_to_base(self, context: Context, read_data: ReadData, control_data: ControlData):
        if context.state_count == 1:
            read_data.scan_flg = True
            read_data.img_flag = False

            start = self.point_to_gird(read_data.localization[:2])
            end = context.base
            t = Thread(target=self._calc_map_path_cb, args=(context, start, end))
            t.start()
        elif context.path:
            return self._follow_path(context, read_data, control_data)
