# Copyright 2024 @With-Robot 3.5
#
# Licensed under the MIT License;
#     https://opensource.org/license/mit

import numpy as np
import matplotlib.pyplot as plt

from util import State, Config, Context, Mission, ReadData, ControlData
from coppeliasim import Coppeliasim
from car import CarClass
from manipulator import ManipulatorClass

from flask import Flask, request
import threading

main = None
app = Flask(__name__)


#
# A class for the entire pick-and-place operation
#
class MainClass:
    def __init__(self):
        self.config = Config()
        self.context = Context()
        with open("mapping.npy", "rb") as f:
            map = np.load(f)
        self.context.map = map >= 0
        self.context.visual_map = self.context.map
        self.context.map_loc = self.init_map_loc(self.context.map.shape)

        self.sim = Coppeliasim()
        self.car = CarClass(self.context.map.shape)
        self.manipulator = ManipulatorClass()

        self.MAP_R, self.MAP_P = np.meshgrid(
            np.linspace(-5, 5, self.context.map.shape[0] + 1),
            np.linspace(-5, 5, self.context.map.shape[1] + 1),
        )
        self.plt_objs = [None] * 100

    def init_map_loc(self, shape):
        map_loc = np.zeros((shape[0], shape[1], 2))
        full = shape[0] * 0.1
        map_loc[:, :, 0] = np.linspace(-full / 2 + 0.05, full / 2 - 0.05, shape[0]).reshape(1, -1)
        full = shape[1] * 0.1
        map_loc[:, :, 1] = np.linspace(-full / 2 + 0.05, full / 2 - 0.05, shape[1]).reshape(-1, 1)
        return map_loc

    def init_mission(self, pick_location, place_location, target):
        if self.context.state == State.StandBy:
            self.context.mission = Mission(pick_location, place_location, target)

    def callback(self, read_data: ReadData, control_data: ControlData):
        self.context.inc_state_counte()
        if self.context.state_count == 1:
            print(f"Enter State: {self.context.state}")

        if self.context.state == State.StandBy:
            if self.context.mission is not None:
                self.context.base = self.car.point_to_gird(read_data.localization[:2])
                self.context.set_state(State.MoveToPick)
        elif self.context.state == State.MoveToPick:
            if self.car.move_to_pick(self.context, read_data, control_data):
                self.context.set_state(State.FindTarget)
        elif self.context.state == State.FindTarget:
            if self.manipulator.find_target(self.context, read_data, control_data):
                self.context.set_state(State.ApproachToTarget)
        elif self.context.state == State.ApproachToTarget:
            if self.manipulator.approach_to_target(self.context, read_data, control_data):
                self.context.set_state(State.PickTarget)
        elif self.context.state == State.PickTarget:
            if self.manipulator.pick_target(self.context, read_data, control_data):
                self.context.set_state(State.MoveToPlace)
        elif self.context.state == State.MoveToPlace:
            if self.car.move_to_place(self.context, read_data, control_data):
                self.context.set_state(State.PlaceTarget)
        elif self.context.state == State.PlaceTarget:
            if self.manipulator.place_target(self.context, read_data, control_data):
                self.context.set_state(State.MoveToBase)
        elif self.context.state == State.MoveToBase:
            if self.car.move_to_base(self.context, read_data, control_data):
                self.context.mission = None
                self.context.set_state(State.StandBy)

        self.visualize(read_data)

    def visualize(self, read_data):
        # remove visual objects
        for i in range(len(self.plt_objs)):
            if self.plt_objs[i] is None:
                break
            self.plt_objs[i].remove()
            self.plt_objs[i] = None

        # map
        map = self.context.map.copy()
        if self.context.path:
            map_path = np.array(self.context.path)
            map[map_path[:, 0], map_path[:, 1]] = 0.5
        # car position
        cp_x, cp_y, _ = read_data.localization[:3]
        _, _, co_z = read_data.localization[3:]
        # lidar positionq
        l_x = cp_x + self.config.lidar_offset * np.cos(co_z)
        l_y = cp_y + self.config.lidar_offset * np.sin(co_z)

        # display image
        self.plt_objs[0] = plt.pcolor(self.MAP_R, self.MAP_P, map * -1, cmap="gray")
        # display car & lidar
        (self.plt_objs[1],) = plt.plot(cp_x, cp_y, color="green", marker="o", markersize=10)
        (self.plt_objs[2],) = plt.plot([cp_x, l_x], [cp_y, l_y], "-b")
        plt.pause(0.001)

    def run(self):
        self.sim.run(self.callback)


@app.route("/stop")
def stop():
    main.sim.run_flag = False
    return


@app.route("/mission", methods=["POST"])
def mission():
    params = request.get_json()
    # set the pick & place locations
    main.init_mission(params["pick_location"], params["place_location"], params["target"])
    return ""


if __name__ == "__main__":
    main = MainClass()

    # start flask web server
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5555, debug=False, use_reloader=False),
        daemon=True,
    ).start()

    # run
    main.run()
