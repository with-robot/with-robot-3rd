import cv2
import numpy as np
import math

from pynput import keyboard
from pynput.keyboard import Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from util import State, Context, Mission, ReadData, ControlData, Config, ConfigObj
from car import get_location, move_to_pick, move_to_place, move_to_base
from manipulator import find_target, pick_target, place_target, approach_to_target


#
# A class for the entire pick-and-place operation
#
class PickAndPlace:
    def __init__(self):
        # coppeliasim simulation instance
        self.sim = RemoteAPIClient().require("sim")
        self.simOMPL = RemoteAPIClient().require("simOMPL")
        # Context of Robot
        self.context = Context()
        # Simulation run flag
        self.run_flag = True

    # key even listener
    def on_press(self, key):
        # Pressing 'a' key will start the mission.
        if key == keyboard.KeyCode.from_char("a"):
            self.context.mission = Mission("A", "B", "Cube")

        # Pressing 'q' key will terminate the simulation
        if key == keyboard.KeyCode.from_char("q"):
            self.run_flag = False

    # init coppeliasim objects
    def init_coppelia(self):
        # robot
        self.youBot = self.sim.getObject("/youBot")
        # reference dummy
        self.youBot_ref = self.sim.getObject("/youBot_ref")
        # collision box
        self.collVolumeHandle = self.sim.getObject("/youBot_coll")
        # Wheel joints
        self.wheels = []
        self.wheels.append(self.sim.getObject("/rollingJoint_fl"))
        self.wheels.append(self.sim.getObject("/rollingJoint_rl"))
        self.wheels.append(self.sim.getObject("/rollingJoint_fr"))
        self.wheels.append(self.sim.getObject("/rollingJoint_rr"))

        # lidar
        self.lidars = []
        for i in range(13):
            self.lidars.append(self.sim.getObject(f"/lidar_{i+1:02d}"))

        # manipulator 5 joints
        self.joints = []
        for i in range(5):
            self.joints.append(self.sim.getObject(f"/youBotArmJoint{i}"))

        # Gripper Joint
        self.joints.append(self.sim.getObject(f"/youBotGripperJoint1"))
        self.joints.append(self.sim.getObject(f"/youBotGripperJoint2"))

        # camera
        self.camera_1 = self.sim.getObject(f"/camera_1")

        # joint & wheel control mode
        self.set_joint_ctrl_mode(self.wheels, self.sim.jointdynctrl_velocity)
        self.set_joint_ctrl_mode(self.wheels, self.sim.jointdynctrl_position)

        # goal locations for path planning
        self.predefined_points = [
            "/bedroom1",
            "/bedroom2",
            "/toilet",
            "/entrance",
            "/dining",
            "/livingroom",
            "/balcony_init",
            "/balcony_end",
        ]
        self.goal_locations = {}
        for goal in range(len(self.predefined_points)):
            self.goal_locations[goal] = self.sim.getObject(goal)

    # set joints dynamic control mode
    def set_joint_ctrl_mode(self, objects, ctrl_mode):
        for obj in objects:
            self.sim.setObjectInt32Param(
                obj, self.sim.jointintparam_dynctrlmode, ctrl_mode
            )

    # read youbot data
    def read_youbot(self, lidar=False, camera=False):
        read_data = ReadData()
        # read localization of youbot
        p = self.sim.getObjectPosition(self.youBot_ref)
        o = self.sim.getObjectQuaternion(self.youBot_ref)
        read_data.localization = np.array(p + o)  # [x,y,z,qw,qx,qy,qz]
        # read H matrix of youbot
        m = self.sim.getObjectMatrix(self.youBot_ref, -1)
        read_data.robot_mat = np.array(m)
        # read wheel joints
        wheels = []
        for wheel in self.wheels:
            theta = self.sim.getJointPosition(wheel)
            wheels.append(theta)
        read_data.wheels = wheels
        # read manipulator joints
        joints = []
        for joint in self.joints:
            theta = self.sim.getJointPosition(joint)
            joints.append(theta)
        read_data.joints = joints
        # read lidar
        if lidar:
            scans = []
            for id in self.lidars:
                scans.append(self.sim.readProximitySensor(id))
            read_data.scans = scans
        # read camera
        if camera:
            result = self.sim.getVisionSensorImg(self.camera_1)
            img = np.frombuffer(result[0], dtype=np.uint8)
            img = img.reshape((result[1][1], result[1][0], 3))
            img = cv2.flip(img, 1)
            read_data.img = img
        # return read_data
        return read_data

    # goal_id : self.context.pick_location_id
    def find_path(self, youbot_data: ReadData, config: Config, goal_id):
        goalPos = self.sim.getObjectPosition(goal_id)
        obstaclesCollection = self.sim.createCollection(0)
        self.sim.addItemToCollection(obstaclesCollection, self.sim.handle_all, -1, 0)
        self.sim.addItemToCollection(
            obstaclesCollection, self.sim.handle_tree, self.youBot, 1
        )
        collPairs = [self.collVolumeHandle, obstaclesCollection]

        search_algo = self.simOMPL.Algorithm.BIRRT
        try:
            # use the path planning state for try-except archi.
            if self.context.path_planning_state:
                task = self.simOMPL.createTask("t")
                self.simOMPL.setAlgorithm(task, search_algo)
                startPos = youbot_data.localization[:3]  # [x,y,z,qw,qx,qy,qz]
                ss = [
                    self.simOMPL.createStateSpace(
                        "2d",
                        self.simOMPL.StateSpaceType.position2d,
                        self.collVolumeHandle,
                        [
                            startPos[0] - config.search_range,
                            startPos[1] - config.search_range,
                        ],
                        [
                            startPos[0] + config.search_range,
                            startPos[1] + config.search_range,
                        ],
                        1,
                    )
                ]
                self.simOMPL.setStateSpace(task, ss)
                self.simOMPL.setCollisionPairs(task, collPairs)
                self.simOMPL.setStartState(task, startPos[:2])
                self.simOMPL.setGoalState(task, goalPos[:2])
                self.simOMPL.setStateValidityCheckingResolution(task, 0.01)
                self.simOMPL.setup(task)

                if self.simOMPL.solve(task, config.search_duration):
                    self.simOMPL.simplifyPath(task, config.search_duration)
                    path = self.simOMPL.getPath(task)

        except ValueError as e:
            print(e)

        if path:
            path_3d = []
            for i in range(0, len(path) // 2):
                path_3d.extend([path[2 * i], path[2 * i + 1], 0.0])
            track_pos_container = self.sim.addDrawingObject(
                self.sim.drawing_spherepoints | self.sim.drawing_cyclic,
                0.02,
                0,
                -1,
                1,
                [1, 0, 1],
            )
            self.context.line_container = track_pos_container
            return path_3d

    # control youbot (return result true if control is finished)
    def control_youbot(self, config: Config, control_data: ControlData):
        control_data.exec_count += 1
        read_data = self.read_youbot(
            lidar=control_data.read_lidar, camera=control_data.read_camera
        )
        read_data.wheels
        result = True
        # path planning @ velocity control mode
        if self.context.state == State.MoveToPick:
            path_3d = self.find_path(goal_id=self.context.pick_location_id)
            goalPos = self.context.pick_location_id
        elif self.context.state == State.MoveToPlace:
            path_3d = self.find_path(goal_id=self.context.place_location_id)
            goalPos = self.context.place_location_id
        # elif self.context.state == State.MoveToBase:
        #     path_3d = self.find_path()

        if control_data.wheels_velocity is not None:
            currPos = read_data.localization[:3]
            pathLength, totalDist = self.sim.getPathLengths(path_3d, 3)
            closet_dist = self.sim.getClosestPosOnPath(path_3d, pathLength, currPos)

            if closet_dist <= config.prev_dist:
                closet_dist += totalDist / 200
            config.prev_dist = closet_dist

            # refine the path smoothly
            targetPoint = self.sim.getPathInterpolateConfig(
                path_3d, pathLength, closet_dist
            )
            self.sim.addDrawingObjectItem(self.context.line_container, targetPoint)

            # calc. the velocity for each of the 4 mecanum wheels
            m = read_data.robot_mat
            m_inv = self.sim.getMatrixInverse(m)
            rel_p = self.sim.multiplyVector(m_inv, targetPoint)
            rel_o = math.atan2(rel_p[1], rel_p[0]) - math.pi / 2

            p_parm = config.drive_parms["p_parm"]
            p_parm_rot = config.drive_parms["p_parm_rot"]
            max_v = config.drive_parms["max_v"]
            max_v_rot = config.drive_parms["max_v_rot"]
            accel_f = config.drive_parms["accel_f"]

            forwback_vel = rel_p[1] * p_parm
            side_vel = rel_p[0] * p_parm
            v = (forwback_vel**2 + side_vel**2) ** 0.5
            if v > max_v:
                forwback_vel *= max_v / v
                side_vel *= max_v / v
            rot_vel = -rel_o * p_parm_rot
            if abs(rot_vel) > max_v_rot:
                rot_vel = max_v_rot * rot_vel / abs(rot_vel)
            prev_forwback_vel = control_data.wheels_velocity_el[0]
            prev_side_vel = control_data.wheels_velocity_el[1]
            prev_rot_vel = control_data.wheels_velocity_el[2]

            df = forwback_vel - prev_forwback_vel
            ds = side_vel - prev_side_vel
            dr = rot_vel - prev_rot_vel

            if abs(df) > max_v * accel_f:
                df = max_v * accel_f * df / abs(df)
            if abs(ds) > max_v * accel_f:
                ds = max_v * accel_f * ds / abs(ds)
            if abs(dr) > max_v_rot * accel_f:
                dr = max_v_rot * accel_f * dr / abs(dr)

            forwback_vel = prev_forwback_vel + df
            side_vel = prev_side_vel + ds
            rot_vel = prev_rot_vel + dr

            # control_data 에 세가지 요소 저장
            control_data.wheels_velocity_el[0] = forwback_vel
            control_data.wheels_velocity_el[1] = side_vel
            control_data.wheels_velocity_el[2] = rot_vel

            # mecanum wheel control
            self.sim.setJointTargetVelocty(
                self.wheels[0], -forwback_vel - side_vel - rot_vel
            )
            self.sim.setJointTargetVelocty(
                self.wheels[1], -forwback_vel + side_vel - rot_vel
            )
            self.sim.setJointTargetVelocty(
                self.wheels[2], -forwback_vel + side_vel + rot_vel
            )
            self.sim.setJointTargetVelocty(
                self.wheels[0], -forwback_vel - side_vel + rot_vel
            )

            if (
                np.linalg.norm(
                    np.array(self.sim.getObjectPosition(goalPos, -1))
                    - np.array(self.sim.getObjectPosition(self.refHandle, -1))
                )
                < 0.6
            ):

                self.sim.removeDrawingObject(self.context.line_container)
                result = True

        if control_data.wheels_position is not None:
            diff_sum = 0
            for i, wheel in enumerate(control_data.wheels_position):
                diff = abs(wheel - read_data.wheels[i])
                diff_sum += diff
                diff = min(diff, control_data.delta)
                if read_data.wheels[i] < wheel:
                    target = read_data.wheels[i] + diff
                else:
                    target = read_data.wheels[i] - diff
                self.sim.setJointTargetPosition(self.wheels[i], target)
            result = diff_sum < 0.005
        if control_data.joints_position is not None:
            diff_sum = 0
            for i, joint in enumerate(control_data.joints_position):
                diff = abs(joint - read_data.joints[i])
                diff_sum += diff
                diff = min(diff, control_data.delta)
                if read_data.joints[i] < joint:
                    target = read_data.joints[i] + diff
                else:
                    target = read_data.joints[i] - diff
                self.sim.setJointTargetPosition(self.joints[i], target)
            result = diff_sum < 0.005
        if control_data.gripper_state is not None:
            p1 = self.sim.getJointPosition(self.joints[-2])
            p2 = self.sim.getJointPosition(self.joints[-1])
            p1 += -0.005 if control_data.gripper_state else 0.005
            p2 += 0.005 if control_data.gripper_state else -0.005
            self.sim.setJointTargetPosition(self.joints[-2], p1)
            self.sim.setJointTargetPosition(self.joints[-1], p2)
            result = control_data.exec_count > 5
        if control_data.control_cb:
            result = control_data.control_cb(self.context, read_data, control_data)
        return result

    # run coppeliasim simulator
    def run_coppelia(self):
        # register a keyboard listener
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()

        # data of control
        control_data = None

        # execution of the simulation
        # using run flag for the loop
        while self.run_flag:
            # callback 함수 실행.
            if control_data:  # if control data exists complete control
                if self.control_youbot(control_data):
                    control_data = None
                self.sim.step()
                continue

            self.context.inc_state_counter()

            if self.context.state == State.StandBy:
                if self.context.mission is not None:
                    self.context.set_state(State.MoveToPick)
                    # get_location(): ReadData.localization
                    base = get_location(self.context, self.read_youbot(lidar=True))
                    self.context.base = base[:3]  # [x,y,z]
            elif self.context.state == State.MoveToPick:
                if self.context.state_counter == 1:
                    self.set_joint_ctrl_mode(
                        self.wheels, self.sim.jointdynctrl_velocity
                    )
                result, control_data = move_to_pick(
                    self.context, self.read_youbot(lidar=True)
                )
                # then move_cb is on the control_cb.
                if result:
                    self.context.set_state(State.FindTarget)
            elif self.context.state == State.FindTarget:
                if self.context.state_counter == 1:
                    self.set_joint_ctrl_mode(
                        self.wheels, self.sim.jointdynctrl_position
                    )
                result, control_data = find_target(
                    self.context, self.read_youbot(camera=True)
                )
                if result:
                    self.context.set_state(State.ApproachToTarget)
            elif self.context.state == State.ApproachToTarget:
                result, control_data = approach_to_target(
                    self.context, self.read_youbot(lidar=True)
                )
                if result:
                    self.context.set_state(State.PickTarget)
            elif self.context.state == State.PickTarget:
                result, control_data = pick_target(
                    self.context, self.read_youbot(camera=True)
                )
                if result:
                    self.context.set_state(State.MoveToPlace)
            elif self.context.state == State.MoveToPlace:
                if self.context.state_counter == 1:
                    self.set_joint_ctrl_mode(
                        self.wheels, self.sim.jointdynctrl_velocity
                    )
                result, control_data = move_to_place(
                    self.context, self.read_youbot(lidar=True)
                )
                if result:
                    self.context.set_state(State.PlaceTarget)
            elif self.context.state == State.PlaceTarget:
                if self.context.state_counter == 1:
                    self.set_joint_ctrl_mode(
                        self.wheels, self.sim.jointdynctrl_position
                    )
                result, control_data = place_target(
                    self.context, self.read_youbot(camera=True)
                )
                if result:
                    self.context.set_state(State.MoveToBase)
            elif self.context.state == State.MoveToBase:
                if self.context.state_counter == 1:
                    self.set_joint_ctrl_mode(
                        self.wheels, self.sim.jointdynctrl_velocity
                    )
                result, control_data = move_to_base(
                    self.context, self.read_youbot(lidar=True)
                )
                if result:
                    self.context.set_state(State.StandBy)
                    self.context.mission = None  # clear mission

            # Run Simulation Step
            self.sim.step()

        # Stop Simulation
        self.sim.stopSimulation()


if __name__ == "__main__":
    client = PickAndPlace()
    client.init_coppelia()
    client.run_coppelia()
